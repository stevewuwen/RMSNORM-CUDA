#!/usr/bin/env python3
"""
NCU 自动化性能分析工具
支持从 ncu-rep 文件直接导入分析

使用方法:
    # 分析 ncu-rep 文件
    python ncu_analyzer.py --import profile.ncu-rep
    
    # 分析并保存报告
    python ncu_analyzer.py --import profile.ncu-rep -o analysis.md
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class Diagnosis:
    bottleneck_type: str
    confidence: str
    reason: str
    suggestions: List[Dict] = field(default_factory=list)


class NCUAnalyzer:
    # 优化建议映射
    OPTIMIZATIONS = {
        "DRAM_MEMORY_BOUND": [
            {
                "name": "Block Tiling",
                "action": "使用共享内存缓存 BM×BK 和 BK×BN 数据块，减少全局内存访问",
                "gain": "3-5x",
                "complexity": "medium",
                "code": "__shared__ float As[BM][BK];\n__shared__ float Bs[BK][BN];"
            },
            {
                "name": "Vectorized Load",
                "action": "使用 float4 加载全局内存，减少内存事务数量",
                "gain": "1.5-2x",
                "complexity": "low",
                "code": "float4 tmp = reinterpret_cast<float4*>(&A[idx])[0];"
            },
            {
                "name": "Prefetching",
                "action": "在计算当前块时预取下一块数据，隐藏内存延迟",
                "gain": "1.2-1.5x",
                "complexity": "high"
            }
        ],
        "L1_PRESSURE_BOUND": [
            {
                "name": "Shared Memory Padding",
                "action": "在共享内存数组第二维添加 +1 padding，避免 bank conflict",
                "gain": "1.2-2x",
                "complexity": "low",
                "code": "__shared__ float As[BM][BK+1];  // +1 padding"
            },
            {
                "name": "Data Transpose",
                "action": "A 矩阵转置存储，改善共享内存访问模式",
                "gain": "1.3-1.8x",
                "complexity": "medium"
            },
            {
                "name": "Fragment Caching",
                "action": "使用寄存器缓存频繁访问的数据片段",
                "gain": "1.2-1.5x",
                "complexity": "medium"
            }
        ],
        "COMPUTE_BOUND": [
            {
                "name": "FMA Optimization",
                "action": "使用 fmaf() 替代 separate mul+add，提高指令吞吐",
                "gain": "1.1-1.3x",
                "complexity": "low",
                "code": "tmp = fmaf(a, b, tmp);  // 替代 tmp += a * b"
            },
            {
                "name": "Loop Unroll",
                "action": "使用 #pragma unroll 展开循环，减少控制开销",
                "gain": "1.1-1.2x",
                "complexity": "low",
                "code": "#pragma unroll"
            },
            {
                "name": "Tensor Core",
                "action": "使用 WMMA 或 mma.sync 指令利用 Tensor Core",
                "gain": "2-8x",
                "complexity": "high"
            }
        ],
        "LATENCY_BOUND": [
            {
                "name": "Double Buffering",
                "action": "使用双缓冲重叠计算和内存访问",
                "gain": "1.2-1.5x",
                "complexity": "medium",
                "code": "__shared__ float As[2][BM*BK]; // ping-pong"
            },
            {
                "name": "Increase Occupancy",
                "action": "调整 block size 或减少寄存器使用",
                "gain": "1.2-2x",
                "complexity": "medium"
            },
            {
                "name": "Warp Tiling",
                "action": "细粒度并行减少同步开销",
                "gain": "1.3-2x",
                "complexity": "high"
            }
        ],
        "OCCUPANCY_BOUND": [
            {
                "name": "Reduce Registers",
                "action": "使用 __launch_bounds__ 或 volatile 减少寄存器",
                "gain": "1.2-2x",
                "complexity": "medium",
                "code": "__launch_bounds__(256, 2)"
            },
            {
                "name": "Adjust TM/TN",
                "action": "减少每个线程处理元素数，降低寄存器压力",
                "gain": "1.2-1.5x",
                "complexity": "low"
            },
            {
                "name": "Dynamic Shared Memory",
                "action": "使用动态共享内存分配",
                "gain": "1.1-1.2x",
                "complexity": "low",
                "code": "extern __shared__ float shared[];"
            }
        ]
    }
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.kernels: Dict[str, Dict[str, float]] = {}
        self.report_file: Optional[str] = None
        
    def extract_from_ncu_rep(self, ncu_rep: str) -> Dict[str, Dict[str, float]]:
        """从 ncu-rep 文件提取所有 kernel 的指标"""
        self.report_file = ncu_rep
        
        cmd = ["ncu", "--import", ncu_rep, "--print-summary", "per-kernel"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"Error running ncu: {result.stderr}")
                return {}
            
            self.kernels = self.parse_summary_output(result.stdout)
            return self.kernels
            
        except Exception as e:
            print(f"Error extracting from ncu-rep: {e}")
            return {}
        
    def parse_summary_output(self, output: str) -> Dict[str, Dict[str, float]]:
        """解析 ncu --print-summary 的输出"""
        kernels = {}
        current_kernel = None
        current_section = None
        
        lines = output.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 识别 kernel 名称行
            # 格式: mysgemm_v1(...) (64, 64, 1)x(32, 32, 1), Device 0, CC 8.9
            if stripped and not stripped.startswith('Section:') and not stripped.startswith('---') and not stripped.startswith('Metric'):
                # 检查是否包含 kernel 特征
                if '(' in stripped and 'x(' in stripped and 'Device' in stripped:
                    # 提取 kernel 名 (在第一个 ( 之前)
                    match = re.match(r'^(.+?)\s*\([^)]+\)\s*\([^)]+\)x\([^)]+\)', stripped)
                    if match:
                        current_kernel = match.group(1).strip()
                        kernels[current_kernel] = {}
                        current_section = None
                        i += 1
                        continue
            
            # 识别 Section
            if stripped.startswith('Section:'):
                current_section = stripped.replace('Section:', '').strip()
                i += 1
                continue
            
            # 解析表格数据行
            if current_kernel and current_section and stripped.startswith('|'):
                # 表格格式: | Metric Name | Unit | Min | Max | Avg |
                parts = [p.strip() for p in stripped.split('|') if p.strip()]
                if len(parts) >= 5:
                    metric_name = parts[0]
                    try:
                        # 使用 Average 列
                        avg_value = float(parts[-1])
                        kernels[current_kernel][metric_name] = avg_value
                    except:
                        pass
                i += 1
                continue
            
            # 解析固定格式行 (格式: Metric Name    Unit    Value)
            if current_kernel and current_section:
                # 尝试匹配: Name    Unit    Min    Max    Avg
                # 或: Name    Unit    Value
                parts = stripped.split()
                if len(parts) >= 3:
                    # 检查最后几个部分是否是数值
                    try:
                        # 尝试解析数值 (可能是 Average 列)
                        for j in range(len(parts) - 1, max(len(parts) - 4, 0), -1):
                            try:
                                value = float(parts[j])
                                # 找到了数值，前面的就是 metric name
                                # 但是需要排除 unit
                                metric_parts = []
                                for k in range(j):
                                    if parts[k] not in ['cycle', '%', 'Ghz', 'ms', 'us', 'ns', 'byte', 'Kbyte', 'Mbyte', 'warp', 'block', 'thread', 'SM']:
                                        metric_parts.append(parts[k])
                                    else:
                                        # 这是 unit，停止
                                        break
                                
                                if metric_parts:
                                    metric_name = ' '.join(metric_parts)
                                    kernels[current_kernel][metric_name] = value
                                break
                            except:
                                continue
                    except:
                        pass
            
            i += 1
        
        return kernels
    
    def get_standardized_metrics(self, kernel_metrics: Dict[str, float]) -> Dict[str, float]:
        """将 NCU 指标名转换为标准指标名"""
        std_metrics = {}
        
        # 映射表 (NCU 原始名 -> 标准名)
        mapping = {
            'Compute (SM) Throughput': 'sm_throughput',
            'Memory Throughput': 'memory_throughput',
            'DRAM Throughput': 'dram_throughput',
            'L1/TEX Cache Throughput': 'l1tex_throughput',
            'L2 Cache Throughput': 'l2_throughput',
            'SM Busy': 'sm_busy',
            'Achieved Occupancy': 'occupancy',
            'Theoretical Occupancy': 'theoretical_occupancy',
            'Duration': 'duration',
            'Block Size': 'block_size',
            'Grid Size': 'grid_size',
            'Registers Per Thread': 'registers',
        }
        
        for ncu_name, std_name in mapping.items():
            if ncu_name in kernel_metrics:
                std_metrics[std_name] = kernel_metrics[ncu_name]
        
        return std_metrics
    
    def get_user_kernel(self) -> Tuple[str, Dict[str, float]]:
        """获取用户 kernel (排除 cuBLAS/cutlass 等库函数)"""
        for name, metrics in self.kernels.items():
            # 排除已知库函数
            if any(lib in name.lower() for lib in ['cublas', 'cutlass', 'cudnn']):
                continue
            # 优先选择包含常见 kernel 命名模式的
            if any(pattern in name for pattern in ['mysgemm', 'kernel', 'matmul', 'softmax']):
                return name, self.get_standardized_metrics(metrics)
        
        # 如果没有匹配的，返回第一个非库函数
        for name, metrics in self.kernels.items():
            if not any(lib in name.lower() for lib in ['cublas', 'cutlass']):
                return name, self.get_standardized_metrics(metrics)
        
        # 返回第一个
        name, metrics = next(iter(self.kernels.items()))
        return name, self.get_standardized_metrics(metrics)
    
    def diagnose(self, metrics: Dict[str, float]) -> List[Diagnosis]:
        """自动诊断瓶颈"""
        diagnoses = []
        
        dram = metrics.get('dram_throughput', 0)
        l1tex = metrics.get('l1tex_throughput', 0)
        sm_busy = metrics.get('sm_busy', 0)
        sm_throughput = metrics.get('sm_throughput', 0)
        memory_throughput = metrics.get('memory_throughput', 0)
        occupancy = metrics.get('occupancy', 0)
        
        # L1 Pressure Bound (最常见)
        if l1tex > 80 and dram < 30:
            diagnoses.append(Diagnosis(
                bottleneck_type="L1_PRESSURE_BOUND",
                confidence="HIGH",
                reason=f"L1/TEX Throughput ({l1tex:.1f}%) > 80%，但 DRAM ({dram:.1f}%) < 30%，说明 L1 缓存压力过高",
                suggestions=self.OPTIMIZATIONS["L1_PRESSURE_BOUND"]
            ))
        
        # DRAM Memory Bound
        if dram > 70:
            diagnoses.append(Diagnosis(
                bottleneck_type="DRAM_MEMORY_BOUND",
                confidence="HIGH",
                reason=f"DRAM Throughput ({dram:.1f}%) > 70%，显存带宽成为瓶颈",
                suggestions=self.OPTIMIZATIONS["DRAM_MEMORY_BOUND"]
            ))
        
        # Compute Bound (SM Throughput 和 Memory Throughput 都很高)
        if sm_throughput > 80 and memory_throughput > 80:
            diagnoses.append(Diagnosis(
                bottleneck_type="COMPUTE_BOUND",
                confidence="MEDIUM",
                reason=f"SM Throughput ({sm_throughput:.1f}%) 和 Memory Throughput ({memory_throughput:.1f}%) 都很高，计算单元接近饱和",
                suggestions=self.OPTIMIZATIONS["COMPUTE_BOUND"]
            ))
        
        # Occupancy Bound
        if occupancy < 50 and occupancy > 0:
            diagnoses.append(Diagnosis(
                bottleneck_type="OCCUPANCY_BOUND",
                confidence="MEDIUM",
                reason=f"Occupancy ({occupancy:.1f}%) < 50%，并行度不足",
                suggestions=self.OPTIMIZATIONS["OCCUPANCY_BOUND"]
            ))
        
        if not diagnoses:
            diagnoses.append(Diagnosis(
                bottleneck_type="UNKNOWN or GOOD",
                confidence="LOW",
                reason="未检测到明显瓶颈，或性能已接近最优",
                suggestions=[]
            ))
        
        return diagnoses
    
    def status_icon(self, value: float, good_threshold: float, bad_threshold: float, 
                    higher_is_better: bool = True) -> str:
        """返回状态图标"""
        if higher_is_better:
            if value >= good_threshold:
                return "✅"
            elif value <= bad_threshold:
                return "❌"
            else:
                return "⚠️"
        else:
            if value <= good_threshold:
                return "✅"
            elif value >= bad_threshold:
                return "❌"
            else:
                return "⚠️"
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """生成分析报告"""
        if not self.kernels:
            return "Error: No kernel data available"
        
        kernel_name, metrics = self.get_user_kernel()
        diagnoses = self.diagnose(metrics)
        
        report = []
        
        # 标题
        report.append("# NCU 自动化性能分析报告")
        report.append("")
        
        # 报告信息
        report.append("## 📁 报告信息")
        report.append(f"- **Kernel**: `{kernel_name}`")
        if self.report_file:
            report.append(f"- **报告文件**: `{self.report_file}`")
        report.append("")
        
        # 执行摘要
        report.append("## 📈 执行摘要")
        report.append("")
        report.append("| 项目 | 数值 |")
        report.append("|------|------|")
        if diagnoses:
            main_diag = diagnoses[0]
            report.append(f"| **主要瓶颈** | {main_diag.bottleneck_type} |")
            report.append(f"| **置信度** | {main_diag.confidence} |")
        
        duration = metrics.get('duration', 0)
        if duration > 0:
            report.append(f"| **执行时间** | {duration:.2f} ms |")
        report.append("")
        
        # 关键指标
        report.append("## 📊 关键指标")
        report.append("")
        report.append("### 性能指标")
        report.append("| 指标 | 数值 | 健康阈值 | 状态 |")
        report.append("|------|------|----------|------|")
        
        sm_busy = metrics.get('sm_busy', 0)
        if sm_busy > 0:
            report.append(f"| SM Busy | {sm_busy:.2f}% | > 70% | {self.status_icon(sm_busy, 70, 40)} |")
        
        sm_throughput = metrics.get('sm_throughput', 0)
        if sm_throughput > 0:
            report.append(f"| SM Throughput | {sm_throughput:.2f}% | > 60% | {self.status_icon(sm_throughput, 60, 30)} |")
        
        memory_throughput = metrics.get('memory_throughput', 0)
        if memory_throughput > 0:
            report.append(f"| Memory Throughput | {memory_throughput:.2f}% | > 60% | {self.status_icon(memory_throughput, 60, 30)} |")
        
        occupancy = metrics.get('occupancy', 0)
        if occupancy > 0:
            report.append(f"| Occupancy | {occupancy:.2f}% | > 50% | {self.status_icon(occupancy, 50, 25)} |")
        
        report.append("")
        report.append("### 内存指标")
        report.append("| 指标 | 数值 | 健康阈值 | 状态 |")
        report.append("|------|------|----------|------|")
        
        dram = metrics.get('dram_throughput', 0)
        if dram > 0:
            report.append(f"| DRAM Throughput | {dram:.2f}% | < 50% | {self.status_icon(dram, 30, 70, False)} |")
        
        l1tex = metrics.get('l1tex_throughput', 0)
        if l1tex > 0:
            report.append(f"| L1/TEX Throughput | {l1tex:.2f}% | < 80% | {self.status_icon(l1tex, 50, 80, False)} |")
        
        l2 = metrics.get('l2_throughput', 0)
        if l2 > 0:
            report.append(f"| L2 Throughput | {l2:.2f}% | < 80% | {self.status_icon(l2, 50, 80, False)} |")
        
        report.append("")
        
        # 配置信息
        report.append("### 配置信息")
        report.append("| 指标 | 数值 |")
        report.append("|------|------|")
        
        block_size = metrics.get('block_size', 0)
        if block_size > 0:
            report.append(f"| Block Size | {int(block_size)} |")
        
        grid_size = metrics.get('grid_size', 0)
        if grid_size > 0:
            report.append(f"| Grid Size | {int(grid_size)} |")
        
        registers = metrics.get('registers', 0)
        if registers > 0:
            report.append(f"| Registers/Thread | {int(registers)} |")
        
        report.append("")
        
        # 诊断详情
        report.append("## 🔍 诊断详情")
        report.append("")
        
        for i, d in enumerate(diagnoses, 1):
            report.append(f"### {i}. {d.bottleneck_type} (置信度: {d.confidence})")
            report.append("")
            report.append(f"**判断依据**: {d.reason}")
            report.append("")
            
            if d.suggestions:
                report.append("**优化建议**:")
                report.append("")
                for j, sug in enumerate(d.suggestions, 1):
                    report.append(f"{j}. **{sug['name']}** (预期收益: {sug['gain']}, 复杂度: {sug['complexity']})")
                    report.append(f"   - {sug['action']}")
                    if 'code' in sug:
                        report.append(f"   - 代码示例:")
                        report.append(f"     ```cpp")
                        for line in sug['code'].split('\n'):
                            report.append(f"     {line}")
                        report.append(f"     ```")
                    report.append("")
        
        # 下一步操作
        report.append("## 🛠️ 下一步操作")
        report.append("")
        report.append("### 建议的 NCU 命令")
        report.append("```bash")
        if self.report_file:
            base_name = Path(self.report_file).stem
            report.append(f"# 优化后重新采集")
            report.append(f"ncu --set full -o {base_name}_optimized --target-processes all ./kernel_optimized")
            report.append("")
            report.append(f"# 查看当前报告详情")
            report.append(f"ncu --import {self.report_file} --page details")
        report.append("```")
        report.append("")
        
        report.append("### 验证清单")
        report.append("- [ ] 实施建议的优化")
        report.append("- [ ] 重新运行 NCU 采集")
        report.append("- [ ] 对比优化前后数据")
        report.append("- [ ] 确认关键指标改善")
        report.append("")
        
        report.append("---")
        report.append("")
        report.append("*报告由 NCU Analyzer 自动生成*")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"✅ 报告已保存到: {output_file}")
        
        return report_text
    
    def print_summary(self):
        """打印摘要到控制台"""
        if not self.kernels:
            print("❌ 未找到 kernel 数据")
            return
        
        kernel_name, metrics = self.get_user_kernel()
        
        print("=" * 60)
        print(f"NCU 性能分析摘要")
        print("=" * 60)
        print(f"Kernel: {kernel_name[:50]}")
        print()
        
        # 关键指标
        key_metrics = [
            ('SM Busy', 'sm_busy', '%'),
            ('SM Throughput', 'sm_throughput', '%'),
            ('Memory Throughput', 'memory_throughput', '%'),
            ('DRAM Throughput', 'dram_throughput', '%'),
            ('L1/TEX Throughput', 'l1tex_throughput', '%'),
            ('Occupancy', 'occupancy', '%'),
        ]
        
        for name, key, unit in key_metrics:
            value = metrics.get(key, 0)
            if value > 0:
                print(f"{name:25s}: {value:8.2f}{unit}")
        
        print()
        print("诊断结果:")
        diagnoses = self.diagnose(metrics)
        for d in diagnoses:
            icon = "🔴" if d.confidence == "HIGH" else ("🟡" if d.confidence == "MEDIUM" else "🟢")
            print(f"  {icon} {d.bottleneck_type} ({d.confidence})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='NCU 自动化性能分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 分析 ncu-rep 文件
    python ncu_analyzer.py --import profile.ncu-rep
    
    # 分析并保存报告
    python ncu_analyzer.py --import profile.ncu-rep -o analysis.md
        """
    )
    parser.add_argument('--import', dest='import_file', help='从 ncu-rep 文件分析')
    parser.add_argument('-o', '--output', help='输出报告文件 (.md)')
    parser.add_argument('--json', action='store_true', help='以 JSON 格式输出指标')
    
    args = parser.parse_args()
    
    analyzer = NCUAnalyzer()
    
    if args.import_file:
        print(f"📥 正在分析 NCU 报告: {args.import_file}")
        kernels = analyzer.extract_from_ncu_rep(args.import_file)
        if not kernels:
            print("❌ 未能从报告中提取数据")
            return
        print(f"✅ 找到 {len(kernels)} 个 kernel")
        # Debug: print raw metrics
        for name, metrics in kernels.items():
            if 'mysgemm' in name.lower() or 'kernel' in name.lower():
                print(f"\n调试 - {name} 的原始指标:")
                for k, v in list(metrics.items())[:10]:
                    print(f"  {k}: {v}")
    else:
        parser.print_help()
        return
    
    if args.json:
        print(json.dumps(analyzer.kernels, indent=2))
    else:
        analyzer.print_summary()
        report = analyzer.generate_report(args.output)
        if not args.output:
            print()
            print(report)


if __name__ == "__main__":
    main()