import torch
import argparse
import os
from itertools import product
import sys

# 尝试导入 vLLM
try:
    from vllm import _custom_ops as ops
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("Warning: vLLM not found. Kernel 4 (vLLM) will not be available.")

sys.path.append('build')
try:
    import rmsnorm_cuda
    HAS_NANOBIND = True
except ImportError:
    HAS_NANOBIND = False
    print("Warning: rmsnorm_cuda not found. Kernel 5 (Custom CUDA) will not be available.")

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    print("Warning: Triton module not found. Kernel 3 (Triton) will not be available.")

kernel_num = 0

# ==========================================
# Kernels 定义
# ==========================================

# Kernel 0: 纯 Python 实现
def pytorch_native_add_rms_norm_func(x, residual, weight, eps=1e-6):
    # In-place 修改 residual 以匹配底层算子逻辑
    residual.add_(x)
    variance = residual.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = residual * torch.rsqrt(variance + eps).to(residual.dtype)
    out = weight * hidden_states
    return out, residual

# Kernel 1: PyTorch 官方 API 实现
def pytorch_official_add_rms_norm_func(x, residual, weight, eps=1e-6):
    residual.add_(x)
    out = torch.nn.functional.rms_norm(residual, (residual.size(-1),), weight, eps)
    return out, residual

# Kernel 2: 编译后的 PyTorch
fast_add_rms_norm = torch.compile(pytorch_official_add_rms_norm_func)
def pytorch_official_compile_add_rms_norm_func(x, residual, weight, eps=1e-6):
    return fast_add_rms_norm(x, residual, weight, eps)

# Kernel 3: Triton 实现
@triton.jit
def triton_add_rms_norm_func(
    X_ptr, Res_ptr, Y_ptr, W_ptr,
    stride_x_row, stride_res_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前行索引
    row_idx = tl.program_id(0)
    
    # 指针偏移计算
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    X_ptr = X_ptr + row_idx * stride_x_row + cols
    Res_ptr = Res_ptr + row_idx * stride_res_row + cols
    Y_ptr = Y_ptr + row_idx * stride_y_row + cols
    W_ptr = W_ptr + cols
    
    # 1. 加载数据并立即转为 FP32
    # 虽然输入是 FP16，但在加载进寄存器后我们用 FP32 存储它们
    x = tl.load(X_ptr, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Res_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # 2. Add-residual (在 FP32 下计算，无溢出风险)
    sum_res = x + res
    
    # 3. 将残差写回内存 (转回 FP16 存储以节省带宽)
    tl.store(Res_ptr, sum_res.to(tl.float16), mask=mask)
    
    # 4. 计算 RMSNorm 方差 (在 FP32 下计算)
    # 计算 sum(x^2)
    sq_sum = sum_res * sum_res
    # 在计算 sum 之前，确保是 fp32 累加
    var = tl.sum(sq_sum, axis=0) / N
    
    # 5. 计算 rsqrt (在 FP32 下，eps 不会被忽略)
    rstd = tl.math.rsqrt(var + eps)
    
    # 6. 加载 Weight (FP16 -> FP32)
    w = tl.load(W_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # 7. 计算输出 (FP32)
    y = sum_res * rstd * w
    
    # 8. 存回最终结果 (FP32 -> FP16)
    tl.store(Y_ptr, y.to(tl.float16), mask=mask)

def triton_add_rms_norm_inference(x, residual, weight, eps=1e-6):
    """
    专门针对推理优化的版本。
    输入 x, residual, weight 均为 FP16 Tensor。
    """
    # 维度检查
    M, N = x.view(-1, x.shape[-1]).shape
    
    # 推理阶段通常 batch 较小，由于是推理，我们直接原位修改 residual 节省内存分配
    y = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # 配置并发度：解码阶段每个行通常很独立，增加 num_warps 提高并行性
    num_warps = 8 if N >= 2048 else 4
    
    grid = (M,)
    
    triton_add_rms_norm_func[grid](
        x, residual, y, weight,
        x.stride(-2), residual.stride(-2), y.stride(-2),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return y, residual

# Kernel 4: vLLM 实现
if HAS_VLLM:
    def add_rms_norm_vllm(x, residual, weight, eps=1e-6):
        """
        注意：vLLM 的 fused_add_rms_norm 原位(in-place)将 x 覆盖为 output，将 residual 覆盖为 x+residual。
        为了和其他 kernel 的非破坏性签名保持一致，这里 clone x。
        但在计时循环中，这会导致多余的 clone 耗时。这里仅作正确性/基线对比。
        """
        out = x.clone()
        ops.fused_add_rms_norm(out, residual, weight, eps)
        return out, residual
else:
    def add_rms_norm_vllm(x, residual, weight, eps=1e-6):
         raise RuntimeError("vLLM is not installed.")

# Kernel 5: 自定义 CUDA 实现
def add_rms_norm_cuda(x, residual, weight, eps=1e-6):
    if not HAS_NANOBIND:
        raise RuntimeError("rmsnorm_cuda not found")
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream
    # 注意：根据你的 C++ 绑定代码，launch_add_rmsnorm_py 只接受 case 1
    rmsnorm_cuda.launch_add_rmsnorm(1, x, residual, weight, out, eps, stream)
    return out, residual

# Kernel 映射表
KERNEL_MAPS = {
    0: ["PyTorch_Pure_Python", pytorch_native_add_rms_norm_func],
    1: ["PyTorch_Official", pytorch_official_add_rms_norm_func],
    2: ["PyTorch_Official_Compile", pytorch_official_compile_add_rms_norm_func],
    3: ["Triton_Custom", triton_add_rms_norm_inference],
    4: ['VLLM_Official', add_rms_norm_vllm],
    5: ["CUDA_Shared_Memory_Add_RMSNorm", add_rms_norm_cuda],
}

# ==========================================
# 验证模块
# ==========================================
# TODO 为什么在tol为1e-3的情况下，在kernel里面转化为fp32的triton算子过不了，为什么没有torch.compile的算子可以过，经过compile后算子就过不了
def verify_correctness(x, residual, weight, tol=1e-2):
    if kernel_num == 0:
        return True  

    # 为了不污染后续测试，做 Deep Copy
    x_ref = x.clone()
    res_ref = residual.clone()
    expected_out, expected_res = KERNEL_MAPS[0][1](x_ref, res_ref, weight)

    x_cus = x.clone()
    res_cus = residual.clone()
    custom_out, custom_res = KERNEL_MAPS[kernel_num][1](x_cus, res_cus, weight)

    # 验证输出一致性
    is_correct_out = torch.allclose(expected_out, custom_out, atol=tol, rtol=tol)
    is_correct_res = torch.allclose(expected_res, custom_res, atol=tol, rtol=tol)
    
    if not (is_correct_out and is_correct_res):
        max_diff_out = torch.max(torch.abs(expected_out - custom_out)).item()
        max_diff_res = torch.max(torch.abs(expected_res - custom_res)).item()
        print(f"[Error] Validation failed for kernel {kernel_num}!")
        print(f"Max Diff Out: {max_diff_out:.4f} | Max Diff Res: {max_diff_res:.4f}")
        
        log_file = "ValidationFailure.txt"
        with open(log_file, "a") as f:
            f.write(f"Validation Failed for Kernel {kernel_num}\n")
            f.write("="*50 + "\n")
            f.write(f"Expected Out (slice):\n{expected_out[0, 0, :10]}\n")
            f.write(f"Custom Out (slice):\n{custom_out[0, 0, :10]}\n\n")
            f.write(f"Expected Res (slice):\n{expected_res[0, 0, :10]}\n")
            f.write(f"Custom Res (slice):\n{custom_res[0, 0, :10]}\n\n")
        print(f"Details written to {log_file}")
        raise RuntimeError("Verification failed. Execution halted.")

    return True

# ==========================================
# 基准测试模块
# ==========================================
def benchmark():
    batch_size_l = [4]
    seq_length_l = [128, 512, 1024, 2048, 4096, 8192]
    hidden_size_l = [4096] 
    
    device = torch.device("cuda:0")
    dtype = torch.float16

    print(f"Running kernel {kernel_num}: {KERNEL_MAPS[kernel_num][0]} on {device}.")
    print("-" * 80)

    for batch_size, seqlen, hidden_size in product(batch_size_l, seq_length_l, hidden_size_l):
        x = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=dtype) * 0.01
        residual = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=dtype) * 0.01
        weight = torch.ones(hidden_size, device=device, dtype=dtype)

        # 1. 正确性验证
        verify_correctness(x, residual, weight)

        # 2. 预热 (Warmup)
        for _ in range(15):
            _ = KERNEL_MAPS[kernel_num][1](x, residual, weight)
        torch.cuda.synchronize()

        # 3. 高精度计时
        iters = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iters):
            _ = KERNEL_MAPS[kernel_num][1](x, residual, weight)
        end_event.record()
        torch.cuda.synchronize()

        # Nsight Compute Profiling 支持
        if seqlen == 1024:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            KERNEL_MAPS[kernel_num][1](x, residual, weight)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        
        avg_time_ms = start_event.elapsed_time(end_event) / iters
        avg_time_s = avg_time_ms / 1000.0
        
        # 4. 计算访存带宽 (GB/s)
        # Add-RMSNorm 带宽消耗分析:
        # Read x: 1x, Read residual: 1x, Write residual: 1x, Read weight: 1x, Write y: 1x
        # 即: 4 份隐层 tensor 大小 + 1 份权重 tensor 大小
        tensor_bytes = x.numel() * x.element_size()
        weight_bytes = weight.numel() * weight.element_size()
        num_bytes = 4 * tensor_bytes + weight_bytes
        
        gbps = (num_bytes / 1e9) / avg_time_s if avg_time_s > 0 else 0

        # 5. 打印结果
        print(f"Shape(B={batch_size}, Seq={seqlen:4d}, Hidden={hidden_size}) | "
              f"Time: {avg_time_ms * 1000:7.2f} µs | "
              f"Bandwidth: {gbps:6.1f} GB/s")

        # 释放显存
        del x, residual, weight
        torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser(description="Add-RMSNorm FP16 Benchmark Framework")
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
    elif args.kernel_num == 3 and not HAS_TRITON:
        print("Error: 选择了 kernel_num 3，但未安装 triton 库。")
    elif args.kernel_num == 4 and not HAS_VLLM:
         print("Error: 选择了 kernel_num 4，但未安装 vllm 库。")
    elif args.kernel_num == 5 and not HAS_NANOBIND:
        print("Error: 选择了 kernel_num 5，但未编译生成 rmsnorm_cuda 模块。")
    else:
        if os.path.exists("ValidationFailure.txt"):
            os.remove("ValidationFailure.txt")
        kernel_num = args.kernel_num
        benchmark()