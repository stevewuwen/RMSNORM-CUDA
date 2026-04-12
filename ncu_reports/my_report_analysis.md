# NCU 自动化性能分析报告

## 📁 报告信息
- **Kernel**: `rms_norm_kernel`
- **报告文件**: `ncu_reports/my_report.ncu-rep`

## 📈 执行摘要

| 项目 | 数值 |
|------|------|
| **主要瓶颈** | DRAM_MEMORY_BOUND |
| **置信度** | HIGH |
| **执行时间** | 144.58 ms |

## 📊 关键指标

### 性能指标
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| SM Throughput | 31.67% | > 60% | ⚠️ |
| Memory Throughput | 82.40% | > 60% | ✅ |
| Occupancy | 91.21% | > 50% | ✅ |

### 内存指标
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| DRAM Throughput | 82.40% | < 50% | ❌ |
| L1/TEX Throughput | 31.87% | < 80% | ✅ |
| L2 Throughput | 28.76% | < 80% | ✅ |

### 配置信息
| 指标 | 数值 |
|------|------|

## 🔍 诊断详情

### 1. DRAM_MEMORY_BOUND (置信度: HIGH)

**判断依据**: DRAM Throughput (82.4%) > 70%，显存带宽成为瓶颈

**优化建议**:

1. **Block Tiling** (预期收益: 3-5x, 复杂度: medium)
   - 使用共享内存缓存 BM×BK 和 BK×BN 数据块，减少全局内存访问
   - 代码示例:
     ```cpp
     __shared__ float As[BM][BK];
     __shared__ float Bs[BK][BN];
     ```

2. **Vectorized Load** (预期收益: 1.5-2x, 复杂度: low)
   - 使用 float4 加载全局内存，减少内存事务数量
   - 代码示例:
     ```cpp
     float4 tmp = reinterpret_cast<float4*>(&A[idx])[0];
     ```

3. **Prefetching** (预期收益: 1.2-1.5x, 复杂度: high)
   - 在计算当前块时预取下一块数据，隐藏内存延迟

## 🛠️ 下一步操作

### 建议的 NCU 命令
```bash
# 优化后重新采集
ncu --set full -o my_report_optimized --target-processes all ./kernel_optimized

# 查看当前报告详情
ncu --import ncu_reports/my_report.ncu-rep --page details
```

### 验证清单
- [ ] 实施建议的优化
- [ ] 重新运行 NCU 采集
- [ ] 对比优化前后数据
- [ ] 确认关键指标改善

---

*报告由 NCU Analyzer 自动生成*