from evaluate import load

# 加载 perplexity 评估模块
perplexity = load("perplexity", module_type="metric")

text1 = """数据反馈非常清晰。Batch Size较小时，1024线程和 512线程拉不开实质差距。但当数据量推到128时，真正的分水岭出现了——1024线程+Pack64由于指针与状态维护的开销叠加，遭遇了严重的寄存器溢出Register Spill。数据被迫溢出到Global Memory，导致耗时突增至9.50 µs。512线程+Pack128精准卡在了SM的物理寄存器容量红线内，耗时降至8.03 µs，这28.2%的性能夺回全是免于Register溢出的功劳。"""

text2 = """在这个前提下，进一步提升单线程的访存吞吐就成了关键。把数据位宽从 64-bit 提至 128-bit（Pack128）后，发射出去的访存指令直接减半，Warp 调度器队列的拥堵立刻得到了缓解。"""
# 准备你要测试的文本列表
texts = [
    text1,text2
]

# 选择一个语言模型（注意：如果是中文文本，一定要选中文模型）
# 这里以一个轻量级的中文 GPT2 为例
model_id = "uer/gpt2-chinese-cluecorpussmall" 

# 计算困惑度
results = perplexity.compute(predictions=texts, model_id=model_id, device='cuda')

print("平均困惑度:", results["mean_perplexity"])
print("每段文本的困惑度:", results["perplexities"])