文件说明：
1. flash_atten.py: 调用baseline+官方实现flash-attention+我实现的attention,在特定的长度下面的性能，结果写入到benchmark_results
2. benchmark_results: 下面存放各个实现的性能数据
2. src/runner.cu: 生成gridDim，blockDim，调用特定的kernel

环境：python3.12, torch 2.8(cuda 12.6), nvcc(cuda 12.6), 显卡驱动(cuda 13.0)
torch: pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
flash_attn: pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl


我正在优化特定情况下的flash-attn，帮我修改下面的入口py文件，该py文件主要充当一个fp16的flash-attn的测试与性能基准框架。
具体来说，它在需要做以下几件事情：
1. 环境与参数初始化：
   - 从命令行接收一个 kernel_num 参数（0-12），用于选择要运行的具体 flash-attn kernel（其中 0 代表调用官方的 flash-attn 库，其余代表不同的自定义优化实现）。
2. 内存分配与数据生成：
   - 在主机端（Host）分配内存，生成随机的单精度浮点数矩阵 Q,K,V。
   - 在设备端（Device/GPU）分配显存，并将初始化好的矩阵数据从主机拷贝到 GPU 显存中。
3. *正确性验证 (Verification)*：
   - 针对不同维度的矩阵大小（128, 256, 512, 1024, 2048, 4096），如果用户选择的不是 0 号 kernel，它会首先运行自定义的 kernel 并将其计算结果与 flash-attn 的标准结果进行对比。
   - 如果结果不一致，程序会报错，并在小矩阵维度下将错误的输出日志记录到 ValidationFailure.txt 中。这保证了各种优化算法计算结果的绝对正确性。
4. *性能基准测试 (Benchmarking)*：
   - 在验证计算正确之后，程序会对所选 kernel 进行多次重复执行（默认 50 次）以进行性能基准测试。
   - 计算出平均执行时间，并由此推算出运算性能（以 GFLOPS 为单位），打印到终端输出。结构如下：
    ```
    Running kernel 0 on device 0.
    dimensions(batch_size=4, seqlen=128, nheads=12, headdim=64)
    Average elapsed time: (0.001218) s, performance: (    3.4) GFLOPS. size: (4*128*12*64).
    dimensions(batch_size=4, seqlen=256, nheads=12, headdim=64)
    Average elapsed time: (0.000009) s, performance: ( 3740.6) GFLOPS. size: (4*256*12*64).
    dimensions(batch_size=4, seqlen=512, nheads=12, headdim=64)
    Average elapsed time: (0.000025) s, performance: (10743.6) GFLOPS. size: (4*512*12*64).
    dimensions(batch_size=4, seqlen=1024, nheads=12, headdim=64)
    Average elapsed time: (0.000126) s, performance: (16981.0) GFLOPS. size: (4*1024*12*64).
    dimensions(batch_size=4, seqlen=2048, nheads=12, headdim=64)
    Average elapsed time: (0.000923) s, performance: (18609.1) GFLOPS. size: (4*2048*12*64).
    dimensions(batch_size=4, seqlen=4096, nheads=12, headdim=64)s
    Average elapsed time: (0.006029) s, performance: (22796.9) GFLOPS. size: (4*4096*12*64).
    ```
5. 清理资源：
   - 测试完成后，释放所有在 Host 和 Device 上申请的内存与句柄资源。
你需要修改的文件如下：
import torch
import math
import time
from flash_attn import flash_attn_func
import argparse

def naive_attention(q, k, v):
    """
    标准的 PyTorch 注意力机制实现 (用于 Baseline 对比)
    输入 shape: (batch_size, seqlen, nheads, headdim)
    """
    # 转换维度以适应 torch.matmul: (batch_size, nheads, seqlen, headdim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Q * K^T / sqrt(d)
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Softmax
    attn = torch.softmax(scores, dim=-1)
    
    # Attn * V
    out = torch.matmul(attn, v)
    
    # 还原维度: (batch_size, seqlen, nheads, headdim)
    out = out.transpose(1, 2).contiguous()
    return out

def benchmark():
    # 测试参数设置
    batch_size = 4
    nheads = 12
    headdim = 64  # head维度通常是 64 或 128
    dtype = torch.float16
    device = "cuda"
    
    # 测试不同的序列长度
    seq_lengths = [128, 512, 1024, 2048, 4096, 8192]
    
    print(f"{'Seq Length':<12} | {'Naive Time (ms)':<16} | {'FlashAttn2 Time (ms)':<20} | {'Speedup':<10}")
    print("-" * 65)

    for seqlen in seq_lengths:
        # FlashAttention-2 要求的默认 shape 是 (batch_size, seqlen, nheads, headdim)
        q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

        # -------------------
        # 1. Warmup (预热)
        # -------------------
        # GPU 测速前必须进行预热，防止初始化开销影响计时
        try:
            for _ in range(10):
                _ = flash_attn_func(q, k, v)
                # 只有在内存允许的情况下才预热 Naive Attention
                if seqlen <= 8192: 
                    _ = naive_attention(q, k, v)
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM during warmup at seqlen {seqlen}. Skipping...")
                continue
            else:
                raise e

        # -------------------
        # 2. 测量 Naive Attention
        # -------------------
        naive_time = float('inf')
        if seqlen <= 8192: # 序列太长时标准注意力会 OOM，所以做了限制
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(50):
                _ = naive_attention(q, k, v)
            torch.cuda.synchronize()
            naive_time = (time.time() - start) / 50 * 1000

        # -------------------
        # 3. 测量 FlashAttention-2
        # -------------------
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            # dropout_p=0.0, causal=False 是默认值
            _ = flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        fa_time = (time.time() - start) / 50 * 1000

        # -------------------
        # 4. 打印结果
        # -------------------
        if seqlen <= 8192:
            speedup = naive_time / fa_time
            print(f"{seqlen:<12} | {naive_time:<16.2f} | {fa_time:<20.2f} | {speedup:.2f}x")
        else:
            print(f"{seqlen:<12} | {'OOM':<16} | {fa_time:<20.2f} | {'N/A':<10}")

def get_args():
    args_parser = argparse.ArgumentParser()
    

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: 该脚本需要可用的 CUDA 环境。")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        benchmark()