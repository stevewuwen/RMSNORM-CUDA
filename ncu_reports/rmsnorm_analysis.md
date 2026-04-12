# NCU 性能分析报告

## 📁 报告信息
- **Kernel**: rms_norm_kernel (id 4)
- **报告文件**: rmsnorm_native_v3.ncu-rep

## 📈 执行摘要

| 项目 | 数值 |
|------|------|
| **主要瓶颈** | DRAM_MEMORY_BOUND |
| **置信度** | HIGH |
| **性能** | ~568 GB/s (接近 A40 的理论带宽 696 GB/s 的 81%) |
| **优化潜力** | 已充分优化 |

## 📊 关键指标 (优化后)

### 性能指标
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| Roofline 性能比 | <20% | > 60% | 正常 (访存密集型) |
| Compute (SM) Throughput | 17.23% | > 70% | 正常 |
| Achieved Occupancy | 89.17% | > 50% | 优秀 |

### 内存指标
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| Memory Throughput | 90.77% | < 50% | 高度利用 |
| DRAM Throughput | 90.77% | < 50% | 高度利用 |

## 🔍 诊断详情

**瓶颈类型**: DRAM_MEMORY_BOUND

**判断依据**:
- DRAM Throughput 高达 90.77%
- 属于典型的 Memory-Bound 操作 (RMSNorm)

## 💡 优化策略

在优化的过程中，我们采用了：
1. **Shared Memory Caching**: 第一遍读取时缓存所有的 input `half`，在乘上 `inv_rms` 时直接从 shared memory 读取，避免了重复的 DRAM 读操作。
2. **Vectorized Load/Store (float4)**: 使用 `float4` 直接读写 `8` 个 `half` 数据，极大提升了 DRAM 吞吐并配合循环展开（Loop Unrolling），进一步提升了指令的并发性。
