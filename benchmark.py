import torch
import argparse
import os
from itertools import product
from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm
from vllm import _custom_ops as ops


HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    print("Warning: Triton module not found. Kernel 2 (Triton) will not be available.")
kernel_num = 0
# Kernel 0: 纯 Python API 实现 (以此作为正确性验证的基准 Gold Standard)
def pytorch_native_rms_norm_func(x, weight, eps=1e-6):
    # 输入形状: x: (..., hidden_size), weight: (hidden_size,)
    # 注意：计算 variance 时最好转为 fp32 防止溢出
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps).to(x.dtype)
    return weight * hidden_states

rms_norm = torch.nn.functional.rms_norm
fast_rms_norm = torch.compile(rms_norm)
# Kernel 1: PyTorch 官方实现 (PyTorch >= 2.4 引入了底层的 RMSNorm)
def pytorch_official_rms_norm_func(x, weight, eps=1e-6):
    return rms_norm(x, (x.size(-1),), weight, eps)

def pytorch_official_compile_rms_norm_func(x, weight, eps=1e-6):
    return fast_rms_norm(x, (x.size(-1),), weight, eps)

# Kernel 2: Triton 实现
if HAS_TRITON:
    @triton.jit
    def triton_rms_norm_fwd_kernel(
        X_ptr, Y_ptr, W_ptr,
        stride_x_row, stride_y_row,
        N, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        # 1D Grid，按照行 (batch * seqlen) 来并行
        row_idx = tl.program_id(0)
        X_ptr = X_ptr + row_idx * stride_x_row
        Y_ptr = Y_ptr + row_idx * stride_y_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # 加载数据，为了精度，内部计算使用 fp32
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # 计算 RMSNorm
        var = tl.sum(x * x, axis=0) / N
        rstd = tl.math.rsqrt(var + eps)
        
        y = x * rstd * w
        
        # 将结果转回原始数据类型并存储
        tl.store(Y_ptr + cols, y.to(X_ptr.dtype.element_ty), mask=mask)

    def triton_rms_norm_func(x, weight, eps=1e-6):
        # 将前两维铺平，统一为 2D 形状处理
        x_2d = x.view(-1, x.shape[-1])
        M, N = x_2d.shape
        y = torch.empty_like(x_2d)

        # Triton block 必须是 2 的幂
        BLOCK_SIZE = triton.next_power_of_2(N)
        
        # 启动 Kernel
        grid = (M,)
        triton_rms_norm_fwd_kernel[grid](
            x_2d, y, weight,
            x_2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return y.view_as(x)
else:
    def triton_rms_norm_func(x, weight, eps=1e-6):
        raise RuntimeError("Triton is not installed.")

import sys
sys.path.append('build')
try:
    import rmsnorm_cuda
    HAS_NANOBIND = True
except ImportError:
    HAS_NANOBIND = False
    print("Warning: rmsnorm_cuda not found. Kernel 3 (Nanobind) will not be available.")

def rms_norm_vllm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    x: [..., hidden_size] (CUDA tensor)
    weight: [hidden_size] (CUDA tensor)
    return: same shape as x
    """
    out = torch.empty_like(x)
    # vLLM fused RMSNorm
    ops.rms_norm(out, x, weight, eps)
    return out

def unsloth_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    使用 Unsloth 的 Triton kernel 计算 RMS LayerNorm
    
    参数:
        x (torch.Tensor): 输入张量，通常 shape 为 [batch_size, seq_len, hidden_dim]
        weight (torch.Tensor): 缩放权重，通常 shape 为 [hidden_dim]
        eps (float): 用于数值稳定的极小值
        
    返回:
        y (torch.Tensor): 归一化后的输出张量
    """
    
    # Fast_RMS_Layernorm 是一个 PyTorch Autograd Function
    # 其 apply 方法的签名为: apply(X, W, eps, gemma)
    # gemma 参数: 如果是 Gemma 架构的模型，设为 True (因为其权重有 +1 偏移)；
    # 对于 Llama, Mistral, Qwen 等绝大多数模型，设为 False。
    gemma = False 
    
    y = Fast_RMS_Layernorm.apply(x, weight, eps, gemma)
    
    return y



def rms_norm_cuda(x, weight, eps=1e-6):
    if not HAS_NANOBIND:
        raise RuntimeError("rmsnorm_cuda not found")
    y = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream
    rmsnorm_cuda.launch_rmsnorm(kernel_num, x, weight, y, eps, stream)
    return y

KERNEL_MAPS = {
    0: ["PyTorch_Pure_Python", pytorch_native_rms_norm_func],
    1: ["PyTorch_Official", pytorch_official_rms_norm_func],
    2: ["PyTorch_Official_Compile", pytorch_official_compile_rms_norm_func],
    3: ["Triton_Custom", triton_rms_norm_func],
    4: ['VLLM_Official', rms_norm_vllm],
    5: ['unsloth_Attention', unsloth_rms_norm],
    6: ["CUDA_Native", rms_norm_cuda],
    7: ["CUDA_Vec8", rms_norm_cuda],
    8: ["CUDA_Shared_Memory", rms_norm_cuda],
}

def verify_correctness(x, weight, tol=1e-3):
    """
    正确性验证模块：使用纯 Python 实现作为标准输出。
    """
    if kernel_num == 0:
        return True  # 自己不用验证自己

    # 获取基准库输出 (Kernel 0)
    expected_out = KERNEL_MAPS[0][1](x, weight)

    # 获取当前选择的 Kernel 输出
    custom_out = KERNEL_MAPS[kernel_num][1](x, weight)
    # 验证一致性
    is_correct = torch.allclose(expected_out, custom_out, atol=tol, rtol=tol)
    if not is_correct:
        max_diff = torch.max(torch.abs(expected_out - custom_out)).item()
        print(f"[Error] Correctness validation failed for kernel {kernel_num}! Max Diff: {max_diff:.4f}")
        
        # 记录详细日志
        log_file = "ValidationFailure.txt"
        with open(log_file, "a") as f:
            f.write(f"Validation Failed for Kernel {kernel_num}\n")
            f.write("="*50 + "\n")
            f.write(f"Expected Output (slice):\n{expected_out[0, 0, :10]}\n\n")
            f.write(f"Custom Output (slice):\n{custom_out[0, 0, :10]}\n\n")
            f.write(f"Max Diff: {max_diff}\n\n")
        print(f"Details written to {log_file}")
        
        raise RuntimeError("Verification failed. Execution halted.")

    return True

def benchmark():
    # LLM 常用的 Shape
    batch_size_l = [4]
    seq_length_l = [128, 512, 1024, 2048, 4096, 8192]
    hidden_size_l = [4096] # 典型模型的 hidden_size, 如 Llama2-7B
    
    device = torch.device("cuda:0")
    dtype = torch.float16

    print(f"Running kernel {kernel_num}: {KERNEL_MAPS[kernel_num][0]} on {device}.")
    print("-" * 80)

    for batch_size, seqlen, hidden_size in product(batch_size_l, seq_length_l, hidden_size_l):
        # 随机生成输入和权重
        x = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=dtype)
        weight = torch.ones(hidden_size, device=device, dtype=dtype)

        # 1. 正确性验证
        verify_correctness(x, weight, kernel_num)

        # 2. 预热 (Warmup)
        for _ in range(15):
            _ = KERNEL_MAPS[kernel_num][1](x, weight)
        torch.cuda.synchronize()

        # 3. 高精度计时
        iters = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iters):
            _ = KERNEL_MAPS[kernel_num][1](x, weight)
        end_event.record()
        torch.cuda.synchronize()

        # 开启分析标记
        if seqlen==1024:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()  # 通知 ncu 开始记录

            KERNEL_MAPS[kernel_num][1](x, weight)

            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()   # 通知 ncu 停止记录
        
        avg_time_ms = start_event.elapsed_time(end_event) / iters
        avg_time_s = avg_time_ms / 1000.0
        
        # 4. 计算访存带宽 Memory Bandwidth (GB/s)
        # RMSNorm 主要是内存读写：读 x，读 weight，写 y。
        # Byte 数 = x的字节数 + weight的字节数 + y的字节数
        num_bytes = x.numel() * x.element_size() + \
                    weight.numel() * weight.element_size() + \
                    x.numel() * x.element_size()
        
        gbps = (num_bytes / 1e9) / avg_time_s if avg_time_s > 0 else 0

        # 5. 打印结果
        print(f"Shape(B={batch_size}, Seq={seqlen:4d}, Hidden={hidden_size}) | "
              f"Time: {avg_time_ms * 1000:7.2f} µs | "
              f"Bandwidth: {gbps:6.1f} GB/s")

        # 释放显存
        del x, weight
        torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser(description="RMSNorm FP16 Benchmark Framework")
    parser.add_argument(
        "kernel_num",
        type=int,
        default=0,
        choices=KERNEL_MAPS.keys(),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if not torch.cuda.is_available():
        print("Error: 该脚本需要可用的 CUDA 环境。")
    elif args.kernel_num == 2 and not HAS_TRITON:
        print("Error: 选择了 kernel_num 2，但未安装 triton 库。")
    else:
        # 清理旧的日志
        if os.path.exists("ValidationFailure.txt"):
            os.remove("ValidationFailure.txt")
        kernel_num = args.kernel_num
        benchmark()