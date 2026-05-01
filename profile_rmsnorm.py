import torch
import torch.cuda.nvtx as nvtx

# 1. 朴素的 RMSNorm 实现（由多个基础算子拼接）
def naive_rmsnorm(x, weight, eps=1e-6):
    # 这里会触发：pow -> mean -> add -> rsqrt -> mul -> mul 多个连续的 Kernel
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return weight * x_normed

def main():
    # 使用较大的 Tensor 以便明显观察到显存读写耗时
    batch_size, seq_len, hidden_dim = 16, 4096, 4096
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float32)
    weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float32)

    # 2. Warmup（预热，排除初始化开销）
    for _ in range(5):
        _ = naive_rmsnorm(x, weight)
    torch.cuda.synchronize()

    # 3. 使用 NVTX 标记，方便在 Profiler 中定位
    nvtx.range_push("Naive_RMSNorm_Block")
    out = naive_rmsnorm(x, weight)
    torch.cuda.synchronize()
    nvtx.range_pop()

if __name__ == "__main__":
    main()