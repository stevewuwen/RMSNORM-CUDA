import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ==========================================
# 0. 中文字体与显示设置 (非常重要)
# ==========================================
# Windows 系统常用 'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
# macOS 系统常用 'Arial Unicode MS' 或 'Songti SC' / 'PingFang SC'
# Linux 系统常用 'WenQuanYi Micro Hei'
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# ==========================================
# 1. 硬件参数设置 (NVIDIA A40)
# ==========================================
PEAK_TFLOPS = 37.4  # 计算瓶颈 (TFLOPS)
BANDWIDTH_GBPS = 696.0  # 显存带宽 (GB/s)
# 机器平衡点 (FLOPs/Byte) = 峰值算力 / 峰值带宽
RIDGE_POINT = (PEAK_TFLOPS * 1000) / BANDWIDTH_GBPS

# ==========================================
# 2. 基础坐标系初始化
# ==========================================
fig, ax = plt.subplots(figsize=(11, 7.5), dpi=150)

# 设置X轴与Y轴范围
x_min, x_max = 1e-1, 1e3
y_min, y_max = 1e-4, 1e2

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel("算术强度 (FLOPs/Byte)", fontsize=12, fontweight="bold")
ax.set_ylabel("性能 (TFLOPS)", fontsize=12, fontweight="bold")
ax.set_title(
    "基于 NVIDIA A40 GPU 的 RMSNorm 算子 Roofline 性能分析",
    fontsize=15,
    pad=15,
    fontweight="bold",
)
ax.grid(True, which="both", ls="--", alpha=0.3)

# ==========================================
# 3. 绘制硬件天际线 (The Roofs)
# ==========================================
x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 500)
# 理论带宽性能 = 算术强度 * 带宽 (将 GFLOPS 转换为 TFLOPS)
y_bandwidth = x_vals * BANDWIDTH_GBPS / 1000.0
# 最终理论上限为带宽高线与计算高线取小值
y_roof = np.minimum(y_bandwidth, PEAK_TFLOPS)

# 绘制红色的 Roofline
ax.plot(x_vals, y_roof, color="red", linewidth=2.5, label="A40 理论天际线")

# 标记计算瓶颈水平线文字
ax.text(
    1e2,
    PEAK_TFLOPS * 1.15,
    f"计算瓶颈: {PEAK_TFLOPS} TFLOPS",
    color="red",
    fontsize=11,
    ha="center",
)

# 标记带宽瓶颈斜线文字
ax.text(
    0.3,
    0.4 * BANDWIDTH_GBPS / 1000,
    f"显存带宽: {BANDWIDTH_GBPS} GB/s",
    color="red",
    fontsize=11,
    rotation=42,
)

# 绘制并标记机器平衡点
ax.scatter(RIDGE_POINT, PEAK_TFLOPS, color="darkred", zorder=5, s=50)
ax.vlines(
    RIDGE_POINT, y_min, PEAK_TFLOPS, color="darkred", linestyle=":", linewidth=1.5
)
ax.text(
    RIDGE_POINT * 1.1,
    1e-3,
    f"机器平衡点\n{RIDGE_POINT:.1f} FLOPs/B",
    color="darkred",
    fontsize=11,
)

# ==========================================
# 4. 绘制 RMSNorm 理论算术强度区间
# ==========================================
intensity_min, intensity_max = 0.67, 1.0

# 绘制垂直阴影带
ax.axvspan(intensity_min, intensity_max, color="royalblue", alpha=0.15, zorder=0)

# 标注阴影带文字
ax.text(
    1.15,
    1e-3,
    "RMSNorm 理论算术强度\n(Memory Bound) 区域",
    color="royalblue",
    fontsize=10,
    verticalalignment="center",
    bbox=dict(
        facecolor="white", edgecolor="royalblue", alpha=0.9, boxstyle="round,pad=0.6"
    ),
)

# ==========================================
# 5. 绘制实际性能轨迹 (随 Batch Size 变化)
# ==========================================
# 模拟的数据点: (Batch_Size, Arithmetic_Intensity, Performance_TFLOPS)
data_points = [
    (1, 0.67, 0.001),  # B=1: 极度延迟受限
    (16, 0.85, 0.05),  # B=16: 爬坡中
    (32, 0.92, 0.15),  # B=32: 爬坡中
    (64, 0.97, 0.65),  # B=64: 触及带宽天花板
    (128, 0.99, 0.68),  # B=128: 完全带宽受限
]

x_data = [pt[1] for pt in data_points]
y_data = [pt[2] for pt in data_points]

# 画出点演变的轨迹（虚线箭头）
ax.plot(
    x_data, y_data, color="teal", linestyle="--", linewidth=1.5, alpha=0.7, zorder=3
)
ax.scatter(
    x_data, y_data, color="teal", s=60, edgecolor="black", zorder=4, label="实测性能"
)

# 注释 起点 B=1
ax.text(
    x_data[0] * 0.9,
    y_data[0] * 0.6,
    "B=1",
    color="teal",
    fontsize=11,
    ha="right",
    fontweight="bold",
)

# 注释 终点 B>=64
ax.text(
    x_data[-1] * 1.05,
    y_data[-1] * 0.8,
    "B $\geq$ 64",
    color="teal",
    fontsize=11,
    ha="left",
    fontweight="bold",
)

# ==========================================
# 6. 添加关键的学术洞察标注
# ==========================================
# 延迟鸿沟 (Latency Gap) 标注
theoretical_y_b1 = x_data[0] * BANDWIDTH_GBPS / 1000.0  # B=1 对应的理论天花板
ax.annotate(
    "",
    xy=(x_data[0], theoretical_y_b1),
    xytext=(x_data[0], y_data[0]),
    arrowprops=dict(arrowstyle="<|-|>", color="orange", lw=2),
    zorder=2,
)
ax.text(
    x_data[0] * 0.85,
    0.008,
    "CPU 下发与 SM 调度开销\n(Latency Bound)",
    color="darkorange",
    fontsize=10,
    ha="right",
    va="center",
)

# 带宽瓶颈 (Bandwidth Bound) 标注
ax.annotate(
    "达到有效物理显存\n(Bandwidth Bound)",
    xy=(x_data[-1], y_data[-1]),
    xytext=(x_data[-1] * 1.5, y_data[-1] * 3.5),
    arrowprops=dict(
        arrowstyle="->", color="green", lw=1.5, connectionstyle="arc3,rad=.2"
    ),
    color="green",
    fontsize=10,
)

# ==========================================
# 7. 图例与布局调整
# ==========================================
legend_elements = [
    Line2D([0], [0], color="red", lw=2.5, label="A40 理论天际线"),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="teal",
        markeredgecolor="black",
        markersize=8,
        label="不同 Batch Size 下的实测性能",
    ),
    Line2D([0], [0], color="teal", linestyle="--", lw=1.5, label="性能演变轨迹"),
    Patch(facecolor="royalblue", alpha=0.15, label="算术强度区间 (0.67 ~ 1.0)"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=10.5)

plt.tight_layout()

# 若需保存图片，可取消下行注释
plt.savefig("assets/图3-1.pdf", bbox_inches="tight")

# plt.show()
