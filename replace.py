import re

with open('thesis_draft.md', 'r') as f:
    text = f.read()

# Abstract replacements
text = text.replace('相较于基于PyTorch基础张量的Python级组合实现（即 `x / torch.sqrt(x.pow(2).mean(-1) + eps) * weight`）', 
                    '相较于基于PyTorch基础张量的Python级组合实现（以下简称 **PyTorch-Python**，即 `x / torch.sqrt(x.pow(2).mean(-1) + eps) * weight`）')
text = text.replace('相较于基于PyTorch内部ATen C++后端的官方算子实现（`torch.nn.functional.rms_norm`）', 
                    '相较于基于PyTorch内部ATen C++后端的官方算子实现（以下简称 **PyTorch-ATen**，`torch.nn.functional.rms_norm`）')

# Chapter 3 replacements
text = text.replace('（此处**“PyTorch原生实现”特指基于PyTorch基础张量操作（Tensor Operations）组合实现的纯Python逻辑**，即 `x / torch.sqrt(x.pow(2).mean(-1) + eps) * weight`）', 
                    '（即 **PyTorch-Python**，`x / torch.sqrt(x.pow(2).mean(-1) + eps) * weight`）')
text = text.replace('基于PyTorch基础张量的Python级组合实现耗时高达 6219.25 µs', 
                    '**PyTorch-Python** 耗时高达 6219.25 µs')
text = text.replace('即使直接调用基于PyTorch内部ATen C++后端的官方算子实现（即 `torch.nn.functional.rms_norm`）', 
                    '即使直接调用 **PyTorch-ATen**')
text = text.replace('**基于PyTorch基础张量的Python级组合实现**的有效带宽约为 **98.1 GB/s**', 
                    '**PyTorch-Python** 的有效带宽约为 **98.1 GB/s**')

# Chapter 5 replacements
text = text.replace('- **基于PyTorch基础张量的Python级组合实现**：采用基于PyTorch基础张量操作（Tensor Operations）组合实现的纯Python逻辑（如`x / torch.sqrt(x.pow(2).mean(-1) + eps) * weight`），耗时 6219.25 µs', 
                    '- **PyTorch-Python**：采用基于PyTorch基础张量操作组合实现的纯Python逻辑（即 `x / torch.sqrt(x.pow(2).mean(-1) + eps) * weight`），耗时 6219.25 µs')
text = text.replace('- **基于PyTorch内部ATen C++后端的官方算子实现（torch.nn.functional.rms_norm）**：', 
                    '- **PyTorch-ATen**（即 `torch.nn.functional.rms_norm`）：')
text = text.replace('无论是**基于PyTorch基础张量的Python级组合实现**，还是**基于PyTorch内部ATen C++后端的官方算子实现**（即 `torch.nn.functional.rms_norm`）', 
                    '无论是 **PyTorch-Python** 还是 **PyTorch-ATen**')
text = text.replace('不仅比**基于PyTorch基础张量的Python级组合实现**快约 **6.8倍**（比**基于PyTorch内部ATen C++后端的官方算子实现**快约 **10.4倍**）', 
                    '不仅比 **PyTorch-Python** 快约 **6.8倍**（比 **PyTorch-ATen** 快约 **10.4倍**）')
text = text.replace('- **基于PyTorch基础张量的Python级组合实现（基线）**：', 
                    '- **PyTorch-Python（基线）**：')
text = text.replace('分别测量基于 PyTorch 内部 ATen C++ 后端的官方半精度（FP16）算子（`torch.nn.functional.rms_norm`）与本文定制的半精度 CUDA 算子（FP16）的计算误差。', 
                    '分别测量 **PyTorch-ATen** 半精度（FP16）算子与本文定制的半精度 CUDA 算子（FP16）的计算误差。')
text = text.replace('| 基于PyTorch内部ATen C++后端的官方算子 (FP16) |', 
                    '| PyTorch-ATen (FP16) |')
text = text.replace('与基于 PyTorch 内部 ATen C++ 后端的官方半精度算子表现出完全一致的极低误差水平。', 
                    '与 **PyTorch-ATen** 半精度算子表现出完全一致的极低误差水平。')

# Chapter 6 replacements
text = text.replace('相较于基于PyTorch内部ATen C++后端的官方算子实现（`torch.nn.functional.rms_norm`）', 
                    '相较于 **PyTorch-ATen**')

with open('thesis_draft.md', 'w') as f:
    f.write(text)
