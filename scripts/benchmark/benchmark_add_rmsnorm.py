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
    print("Warning: rmsnorm_cuda not found. Kernel 3-6 (Nanobind) will not be available.")

kernel_num = 0

def pytorch_native_add_rms_norm_func(x, residual, weight, eps=1e-6):
    residual_out = x + residual
    variance = residual_out.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = residual_out * torch.rsqrt(variance + eps).to(residual_out.dtype)
    return weight * hidden_states, residual_out

rms_norm = torch.nn.functional.rms_norm

def pytorch_official_add_rms_norm_func(x, residual, weight, eps=1e-6):
    residual += x
    out = rms_norm(residual, (residual.size(-1),), weight, eps)
    return out, residual

fast_add_rms_norm = torch.compile(pytorch_official_add_rms_norm_func)
def pytorch_official_compile_add_rms_norm_func(x, residual, weight, eps=1e-6):
    return fast_add_rms_norm(x, residual, weight, eps)

def add_rms_norm_fusion_cuda(x, residual, weight, eps=1e-6):
    if not HAS_NANOBIND:
        raise RuntimeError("rmsnorm_cuda not found")
    y = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream
    rmsnorm_cuda.launch_add_rmsnorm(1, x, residual, weight, y, eps, stream)
    return y, residual

def add_rms_norm_not_fusion_cuda(x, residual, weight, eps=1e-6):
    if not HAS_NANOBIND:
        raise RuntimeError("rmsnorm_cuda not found")
    residual += x
    y = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream
    rmsnorm_cuda.launch_rmsnorm(6, residual, weight, y, eps, stream)
    return y, residual

KERNEL_MAPS = {
    7: ["PyTorch_Pure_Python", pytorch_native_add_rms_norm_func],
    8: ["PyTorch_Official", pytorch_official_add_rms_norm_func],
    0: ["PyTorch_Official_Compile", pytorch_official_compile_add_rms_norm_func],
    1: ["CUDA_NOT_FUSION_TLP", add_rms_norm_not_fusion_cuda],
    2: ["CUDA_FUSION_TLP", add_rms_norm_fusion_cuda],
}

def verify_correctness(x, residual, weight, tol=1e-2):
    if kernel_num == 0:
        return True 

    res_clone_0 = residual.clone()
    expected_out, expected_res = KERNEL_MAPS[0][1](x, res_clone_0, weight)
    
    res_clone_k = residual.clone()
    custom_out, custom_res = KERNEL_MAPS[kernel_num][1](x, res_clone_k, weight)
    
    is_out_correct = torch.allclose(expected_out, custom_out, atol=tol, rtol=tol)
    is_res_correct = torch.allclose(expected_res, custom_res, atol=tol, rtol=tol)
    
    if not (is_out_correct and is_res_correct):
        max_diff_out = torch.max(torch.abs(expected_out - custom_out)).item()
        max_diff_res = torch.max(torch.abs(expected_res - custom_res)).item()
        
        print(f"[Error] Correctness validation failed for kernel {kernel_num}!")
        print(f"Max Diff Out: {max_diff_out:.4f} | Max Diff Res: {max_diff_res:.4f}")
        
        log_file = "ValidationFailure.txt"
        with open(log_file, "a") as f:
            f.write(f"Validation Failed for Kernel {kernel_num}\n")
            f.write("="*50 + "\n")
            f.write(f"Expected Output (slice):\n{expected_out[0, 0, :10]}\n\n")
            f.write(f"Custom Output (slice):\n{custom_out[0, 0, :10]}\n\n")
            f.write(f"Expected Res (slice):\n{expected_res[0, 0, :10]}\n\n")
            f.write(f"Custom Res (slice):\n{custom_res[0, 0, :10]}\n\n")
            f.write(f"Max Diff Out: {max_diff_out} | Max Diff Res: {max_diff_res}\n\n")
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
        residual = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=dtype)
        weight = torch.ones(hidden_size, device=device, dtype=dtype)

        # 1. 正确性验证
        verify_correctness(x, residual, weight)

        # 2. 预热 (Warmup: 使得 GPU 达到稳态，并完成任何 JIT 编译)
        for _ in range(15):
            # 由于 residual 是原地修改的，每次循环都使用其拷贝，以防数据溢出
            res_dummy = residual.clone()
            _ = KERNEL_MAPS[kernel_num][1](x, res_dummy, weight)
        torch.cuda.synchronize()

        # 3. 高精度 NCU Profiling
        print(f"Dispatching Shape(B={batch_size}, Seq={seqlen:4d}, Hidden={hidden_size}) for NCU...")
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()  # 通知 NCU 开始记录
        
        # 添加 NVTX Marker，NCU 中会显示这个名称，方便区分不同的 Shape
        nvtx.range_push(f"Shape_B{batch_size}_S{seqlen}_H{hidden_size}")

        res_ncu = residual.clone()  # 同样克隆一个作为分析用的真实数据
        KERNEL_MAPS[kernel_num][1](x, res_ncu, weight)

        nvtx.range_pop()
        
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()   # 通知 NCU 停止记录

        # 释放显存
        del x, residual, weight, res_ncu
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
    else:
        if os.path.exists("ValidationFailure.txt"):
            os.remove("ValidationFailure.txt")
        kernel_num = args.kernel_num
        benchmark()