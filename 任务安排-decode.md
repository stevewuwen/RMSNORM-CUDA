### RMSNorm

RMSNorm 的输入张量形状标准就是：(batch, seq_len, dim)

(batch_size, sequence_length, hidden_size)
  ↑            ↑                ↑
批次大小     序列长度        特征维度（dim）
1. batch_size：一次喂多少条数据
2. sequence_length：句子 /token 的长度
3. hidden_size：每个 token 的向量维度（比如 Qwen 里的 1024、2048、4096）

RMSNorm只在最后一维 dim 上做归一化！也就是对每个 token 自己的特征向量做归一化。
公式简化理解：
对每个 token: [d1, d2, ..., d_dim] → 做 RMSNorm

variance = hidden_states.pow(2).mean(-1, keepdim=True)
-1 就代表最后一维 dim。

### 核心技术路线与优化点

你的优化应该是一个**循序渐进（Step-by-Step）**的过程，这也是论文最好的叙事结构。

#### Step 1: Baseline 搭建
*   实现 PyTorch 纯 Python API 拼接的 RMSNorm。
*   获取 PyTorch 官方/Triton 实现的 RMSNorm 作为进阶 Baseline。

#### Step 2: 朴素 CUDA 实现 (Naive Kernel)
*   **逻辑：** 经典的两阶段。先启动一个 Kernel 计算每个 token 的平方和，再启动一个 Kernel 进行归一化。
*   **缺点：** 两次 Kernel Launch，读取了两次输入数据。

#### Step 3: 单 Kernel 优化 (One-Pass RMSNorm)
*   让一个 Thread Block 处理一个/多个 Token（行）。
*   在 Block 内部使用 Shared Memory 或 Warp Primitives（`__shfl_down_sync`）进行并行规约（Parallel Reduction），求出方差。
*   求出方差后，直接在同一个 Kernel 里完成归一化和输出。

#### Step 4: 访存极致优化 (Memory Access Optimization) - **核心**
*   **向量化访存 (Vectorized Access)：** 使用 `float4` (FP32) 或 `int4` (如果做 FP16/BF16) 读取数据，将显存请求合并，大幅提高带宽利用率。
*   **合并访存 (Coalesced Memory Access)：** 确保同一个 Warp 里的线程读取连续的内存地址。

#### Step 5: 计算与指令优化
*   **Fast Math：** 使用 CUDA 内部的高速数学函数，例如用 `rsqrtf()` 代替 `1.0 / sqrt()`。
*   **混合精度：** 输入输出为 FP16/BF16，但在计算平方和时使用 FP32 累加，防止精度溢出（这在 LLM 推理中极为重要，可作为论文的一个亮点）。

#### Step 6: 算子融合设计 (Operator Fusion)
*   **Fused Add-RMSNorm：**
    *   大模型结构：`x = x + attention_out`，然后 `y = RMSNorm(x)`。
    *   如果不融合，需要把 `x` 写回显存，再读出来做 RMSNorm。
    *   **你的融合逻辑：** Kernel 同时读入 `x` 和 `attention_out`，在寄存器中完成加法，顺便计算平方和，然后做 RMSNorm，最后把 `y` (Norm结果) 和 `x` (加法结果，供下一层用) 写回显存。访存量锐减！

---

### 三、 论文工作与时间路线图 (以 4 个月为例)

#### 第 1 个月：环境、理论与基线
1.  **环境配置：** 找一台带有 NVIDIA GPU（最好是 Ampere 架构如 RTX 3090 / A10 / A100）的机器。
2.  **工具学习：** 学习使用 **Nsight Compute (NCU)**。这是你论文的关键，你需要用 NCU 的截图（如 Memory Bandwidth 占比、Roofline 图）来充实论文。
3.  **跑通基线：** 编写测试框架，对比 PyTorch 原生 RMSNorm 的耗时，记录数据。
4.  建立rms_norm的Roofline模型，确立性能基线。

#### 第 2 个月：CUDA 核心优化开发
1.  实现 Block-level reduction 的单 Kernel RMSNorm。
2.  加入 Warp-level reduction（使用 `__shfl_down_sync`）。
3.  加入 `float4` 向量化访存。
4.  **里程碑：** 你的 CUDA 版本应该在耗时上明显击败 PyTorch 原生版本，接近或达到理论显存带宽的 80% 以上。

#### 第 3 个月：算子融合与进阶 (拉开差距的部分)
1.  实现 `Fused Add-RMSNorm` 算子。
2.  针对主流大模型的 Hidden Size（如 LLaMA-7B 的 4096，或者 8192）进行特化调优（Tuning）。比如，当维度是 4096 时，刚好用 4 个 Warp (128 threads)，每个 thread 用 float4 处理 8 个 FP16 元素。
3.  对比测试：对比 `PyTorch Add + PyTorch RMSNorm` 与你的 `Fused Kernel` 的性能差异。

#### 第 4 个月：测试、图表与论文撰写
1.  **实验设计：** 在不同的 Batch Size（推理阶段通常较小如 1~32）和不同的 Sequence Length 下进行全面 Benchmark。
2.  **核心图表绘制：**
    *   加速比柱状图 (Speedup vs Baseline)。
    *   显存带宽利用率折线图 (Achieved Memory Bandwidth)。
    *   Roofline Model 图（Nsight Compute 直接生成）。
3.  开始撰写论文。

---

### 实验对比设计 (你的论文数据从哪来？)

为了让你的论文看起来具有高水平的学术和工程价值，你需要设计严谨的对比实验。

**Baseline 对比阵营：**
1.  **Baseline 1 (小白):** `torch.mean` + `torch.sqrt` 纯算子拼接。
2.  **Baseline 2 (进阶):** PyTorch `torch.nn.RMSNorm` (如果有的话，或者相关库原生API)。
3.  **Baseline 3 (工业界):** 引入 `Triton` 语言写的 RMSNorm (Triton 写 RMSNorm 只有 10 行代码，性能极好，可以作为你手写 CUDA 挑战的终极 Boss)。
4.  **Baseline 4 (融合对比):** `torch.add` + `torch.nn.RMSNorm` vs `Your Fused Add-RMSNorm`。

**评测指标 (Metrics)：**
1.  **Latency (延迟):** 执行时间的绝对值（微秒 $\mu s$）。
2.  **Memory Throughput (显存吞吐量):** GB/s，衡量你有没有把显存带宽榨干。
3.  **Numerical Error (数值误差):** 比较你的输出与 PyTorch FP32 输出的 Max Error (最大绝对误差)，证明你的算子结果是对的（且使用了 FP32 累加保证了精度）。

### 五、 给你的建议

1.  **多用 Nsight Compute (NCU)：** 导师和答辩评委最喜欢看 NCU 的 Profiling 截图。不要只贴时间数字，要贴图分析“我的 Kernel 提升是因为 Global Memory Read/Write 的次数减少了”。
2.  **参考开源代码：** 不要闭门造车。强烈建议阅读 **vLLM** 或者 **FasterTransformer** 源码中的 `rmsnorm_kernels.cu`。你可以参考他们的思路，然后在论文中用自己的语言解析，并复现出来。
3.  **控制变量法：** 在做 Benchmark 时，记得提前做 `Warmup`，并且使用 `cudaEventRecord` 或者合适的 CUDA timer 来精确测量微秒级别的耗时。