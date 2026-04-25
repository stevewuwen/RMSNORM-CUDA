import torch
import argparse
import os
from itertools import product
from vllm import _custom_ops as ops
import sys
import torch.cuda.nvtx as nvtx # 引入 nvtx 用于在 NCU 中标记范围

sys.path.append('build')
try:
    import rmsnorm_cuda
    HAS_NANOBIND = True
except ImportError:
    HAS_NANOBIND = False
    print("Warning: rmsnorm_cuda not found. Kernel 3 (Nanobind) will not be available.")

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    print("Warning: Triton module not found. Kernel 2 (Triton) will not be available.")

kernel_num = 0

# Kernel 0: 纯python实现
def pytorch_native_rms_norm_func(x, weight, eps=1e-6):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps).to(x.dtype)
    return weight * hidden_states

rms_norm = torch.nn.functional.rms_norm
fast_rms_norm = torch.compile(rms_norm)

# Kernel 1: pytorch官方实现
def pytorch_official_rms_norm_func(x, weight, eps=1e-6):
    return rms_norm(x, (x.size(-1),), weight, eps)

# kernel 2： 编译后的pytorch
def pytorch_official_compile_rms_norm_func(x, weight, eps=1e-6):
    return fast_rms_norm(x, (x.size(-1),), weight, eps)

# Kernel 3: Triton 实现（小 batch_size 时性能稍差）
if HAS_TRITON:
    @triton.jit
    def triton_rms_norm_fwd_kernel(
        X_ptr, Y_ptr, W_ptr,
        stride_x_row, stride_y_row,
        N, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)

        X_ptr = X_ptr + row_idx * stride_x_row
        Y_ptr = Y_ptr + row_idx * stride_y_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_sq = x * x
        var = tl.sum(x_sq, axis=0) / N

        rstd = tl.math.rsqrt(var + eps)

        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w

        tl.store(
            Y_ptr + cols,
            y.to(X_ptr.dtype.element_ty),
            mask=mask
        )

    def triton_rms_norm_func(x, weight, eps=1e-6):
        x_2d = x.view(-1, x.shape[-1])
        M, N = x_2d.shape

        y = torch.empty_like(x_2d)
        BLOCK_SIZE = max(1024, triton.next_power_of_2(N))

        grid = (M,)

        triton_rms_norm_fwd_kernel[grid](
            x_2d,
            y,
            weight,
            x_2d.stride(0),
            y.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8
        )

        return y.view_as(x)

else:
    def triton_rms_norm_func(x, weight, eps=1e-6):
        raise RuntimeError("Triton is not installed.")

def rms_norm_vllm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, eps)
    return out

def rms_norm_cuda(x, weight, eps=1e-6):
    if not HAS_NANOBIND:
        raise RuntimeError("rmsnorm_cuda not found")
    y = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream
    rmsnorm_cuda.launch_rmsnorm(kernel_num, x, weight, y, eps, stream)
    return y

KERNEL_MAPS = {
    8: ["PyTorch_Pure_Python", pytorch_native_rms_norm_func],
    9: ["PyTorch_Official", pytorch_official_rms_norm_func],
    0: ["PyTorch_Official_Compile", pytorch_official_compile_rms_norm_func],
    1: ["Triton_Custom", triton_rms_norm_func],
    2: ['VLLM_Official', rms_norm_vllm],
    3: ["CUDA_Native", rms_norm_cuda],
    4: ["CUDA_Vec8", rms_norm_cuda],
    5: ["CUDA_Shared_Memory", rms_norm_cuda],
    6: ["CUDA_ILP", rms_norm_cuda],
    7: ["CUDA_Pack128", rms_norm_cuda],
}

def verify_correctness(x, weight, tol=1e-3):
    if kernel_num == 0:
        return True 

    expected_out = KERNEL_MAPS[0][1](x, weight)
    custom_out = KERNEL_MAPS[kernel_num][1](x, weight)
    
    is_correct = torch.allclose(expected_out, custom_out, atol=tol, rtol=tol)
    if not is_correct:
        max_diff = torch.max(torch.abs(expected_out - custom_out)).item()
        print(f"[Error] Correctness validation failed for kernel {kernel_num}! Max Diff: {max_diff:.4f}")
        
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
    batch_size_l = [1, 2, 4, 8, 16, 32, 64, 128]
    seq_length_l = [1]
    hidden_size_l = [4096] 
    
    device = torch.device("cuda:0")
    dtype = torch.float16

    print(f"Running kernel {kernel_num}: {KERNEL_MAPS[kernel_num][0]} on {device}.")
    print("Wait for NCU output for exact GPU duration and bandwidth...")
    print("-" * 80)

    for batch_size, seqlen, hidden_size in product(batch_size_l, seq_length_l, hidden_size_l):
        x = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=dtype)
        weight = torch.ones(hidden_size, device=device, dtype=dtype)

        # 1. 正确性验证 (修复了原先传入 kernel_num 污染 tol 默认值的 Bug)
        verify_correctness(x, weight)

        # 2. 预热 (Warmup: 使得 GPU 达到稳态，并完成任何 JIT 编译)
        for _ in range(15):
            _ = KERNEL_MAPS[kernel_num][1](x, weight)
        torch.cuda.synchronize()

        # 3. 高精度 NCU Profiling
        print(f"Dispatching Shape(B={batch_size}, Seq={seqlen:4d}, Hidden={hidden_size}) for NCU...")
        
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()  # 通知 NCU 开始记录
        
        # 添加 NVTX Marker，NCU 中会显示这个名称，方便区分不同的 Shape
        nvtx.range_push(f"Shape_B{batch_size}_S{seqlen}_H{hidden_size}")

        # 对于 NCU 分析，只需要执行 1 次！
        # NCU 会拦截底层的 Kernel launch，并为了采样数据而在底层自动重放(Replay)该 Kernel。
        # 如果这里用循环，会导致 NCU 跑极其漫长的时间。
        KERNEL_MAPS[kernel_num][1](x, weight)

        nvtx.range_pop()
        
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()   # 通知 NCU 停止记录

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
        if os.path.exists("ValidationFailure.txt"):
            os.remove("ValidationFailure.txt")
        kernel_num = args.kernel_num
        benchmark()