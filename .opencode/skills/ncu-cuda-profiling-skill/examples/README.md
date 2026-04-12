# NCU CUDA Profiling Skill - 示例和工具

本目录包含自动化脚本和示例，帮助你快速上手 NCU 性能分析。

## 📁 文件说明

| 文件              | 说明              | 用法                                             |
| ----------------- | ----------------- | ------------------------------------------------ |
| `auto_profile.sh` | 自动化分析脚本    | `./auto_profile.sh ./kernel report_name`         |
| `ncu_analyzer.py` | Python 深度分析器 | `python ncu_analyzer.py --import report.ncu-rep` |

---

## 🚀 快速开始

### 1. 自动化分析 (auto_profile.sh)

```bash
# 基础用法
./auto_profile.sh ../your_cuda_project/matmul my_analysis

# 高级选项
./auto_profile.sh ./kernel report_name --detailed --export-csv
```

**功能**:
- 自动运行 NCU 完整采集
- 生成 Markdown 分析报告
- 导出 CSV 指标数据
- 自动诊断瓶颈类型

**输出**:
```
report_name/
├── report_name.ncu-rep      # NCU 原始报告
├── report_name.csv          # CSV 指标
├── report_name_analysis.md  # 分析报告
└── summary.txt              # 执行摘要
```

### 2. Python 深度分析 (ncu_analyzer.py)

```bash
# 分析已有报告
python ncu_analyzer.py --import my_report.ncu-rep

# 生成可视化图表（需要 matplotlib）
python ncu_analyzer.py --import my_report.ncu-rep --plot

# 对比两个报告
python ncu_analyzer.py --diff report1.ncu-rep report2.ncu-rep
```

**依赖**:
```bash
pip install pandas matplotlib numpy
```

---

## 📊 实际案例分析

### 案例 1: 矩阵乘法优化

```bash
# 优化前
cd your_project
./auto_profile.sh ./matmul_before before

# 优化后 (添加 shared memory tiling)
./auto_profile.sh ./matmul_after after

# 对比
python ncu_analyzer.py --diff before.ncu-rep after.ncu-rep
```

**预期输出**:
```
性能对比:
- 执行时间: 1200μs -> 340μs (3.5x 提升)
- L1 Hit Rate: 2% -> 78%
- DRAM Throughput: 85% -> 25%
```

### 案例 2: 定位 Bank Conflict

```bash
# 分析 kernel
./auto_profile.sh ./kernel kernel_analysis

# 检查报告中 Shared Memory 相关指标
# 如果 L1/TEX Throughput 高但 L1 Hit Rate 低，可能存在 bank conflict
```

---

## 🔧 自定义脚本

你可以基于 `auto_profile.sh` 创建自己的分析流程：

```bash
#!/bin/bash
# my_custom_profile.sh

KERNEL=$1
REPORT=$2

# 1. 运行 NCU
ncu --set full -o $REPORT --target-processes all $KERNEL

# 2. 提取关键指标
ncu --import $REPORT.ncu-rep --print-summary per-kernel > summary.txt

# 3. 自定义分析
python3 << EOF
import json
# 你的分析逻辑
EOF

# 4. 生成报告
echo "分析完成！"
```

---

## 💡 最佳实践

1. **多次运行取平均**
   ```bash
   for i in {1..3}; do
       ./auto_profile.sh ./kernel run_$i
   done
   ```

2. **Warmup 很重要**
   - 确保 kernel 先运行几次再采集
   - 避免冷启动影响

3. **控制变量**
   - 每次只改一处，便于定位问题
   - 使用 `--diff` 对比版本

4. **关注 Roofline**
   - Roofline 比 > 60% 才算优化到位
   - 不要只看单一指标

---

## 📚 参考

- [SKILL.md](../SKILL.md) - 完整诊断规则
- [NCU 官方文档](https://docs.nvidia.com/nsight-compute/)
- [CUDA 性能优化指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)