with open('thesis_draft.md', 'r') as f:
    text = f.read()

text = text.replace('相较于基于PyTorch基础张量操作组合的RMSNorm原生实现，本文优化的RMSNorm单算子执行速度提升约6.8倍（相比新版官方实现提速逾10倍）',
                    '相较于 **PyTorch-Python**，本文优化的RMSNorm单算子执行速度提升约6.8倍（相比 **PyTorch-ATen** 提速逾10倍）')

text = text.replace('相较于基于PyTorch内部ATen C++后端的官方算子实现（以下简称 **PyTorch-ATen**，`torch.nn.functional.rms_norm`）达到了10倍',
                    '相较于 **PyTorch-ATen** 达到了10倍')

with open('thesis_draft.md', 'w') as f:
    f.write(text)
