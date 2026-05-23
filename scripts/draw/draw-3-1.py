import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ==========================================
# 0. 中文字体与显示设置
# ==========================================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 字体整体仍保持较大，但对不同元素分级，避免互相挤压
FONT_SCALE = 2

FS_LABEL = int(12 * FONT_SCALE)
FS_TEXT = int(10.5 * FONT_SCALE)
FS_SMALL = int(9.5 * FONT_SCALE)
FS_TICK = int(10 * FONT_SCALE)
FS_LEGEND = int(9.6 * FONT_SCALE)

# ==========================================
# 1. 硬件参数设置 NVIDIA A40
# ==========================================
PEAK_TFLOPS = 37.4
BANDWIDTH_GBPS = 696.0
RIDGE_POINT = (PEAK_TFLOPS * 1000) / BANDWIDTH_GBPS

# ==========================================
# 2. 基础坐标系初始化
# ==========================================
fig, ax = plt.subplots(figsize=(18.5, 10.2), dpi=180)

x_min, x_max = 1e-1, 1e3
y_min, y_max = 1e-4, 2e2

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel(
    "算术强度 (FLOPs/Byte)",
    fontsize=FS_LABEL,
    fontweight="bold",
    labelpad=12,
)
ax.set_ylabel(
    "性能 (TFLOPS)",
    fontsize=FS_LABEL,
    fontweight="bold",
    labelpad=12,
)

ax.tick_params(axis="both", which="major", labelsize=FS_TICK, width=1.6, length=8)
ax.tick_params(axis="both", which="minor", width=1.1, length=4)

ax.grid(True, which="both", ls="--", alpha=0.28)

# ==========================================
# 3. 绘制硬件天际线
# ==========================================
x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 500)
y_bandwidth = x_vals * BANDWIDTH_GBPS / 1000.0
y_roof = np.minimum(y_bandwidth, PEAK_TFLOPS)

ax.plot(
    x_vals,
    y_roof,
    color="red",
    linewidth=4.2,
    label="A40 理论天际线",
    zorder=2,
)

# 计算瓶颈文字：右移并上移，避免压到红线
ax.text(
    260,
    PEAK_TFLOPS * 1.22,
    f"计算瓶颈: {PEAK_TFLOPS} TFLOPS",
    color="red",
    fontsize=FS_TEXT,
    ha="center",
    va="bottom",
)

# 显存带宽标注：放到红线左下侧，并加白底避免网格干扰
ax.text(
    0.22,
    0.5,
    f"显存带宽: {BANDWIDTH_GBPS} GB/s",
    color="red",
    fontsize=FS_TEXT,
    rotation=16,
    ha="left",
    va="center",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.9,
        pad=0.28,
    ),
    zorder=5,
)

# 机器平衡点
ax.scatter(
    RIDGE_POINT,
    PEAK_TFLOPS,
    color="darkred",
    zorder=5,
    s=120,
)

ax.vlines(
    RIDGE_POINT,
    y_min,
    PEAK_TFLOPS,
    color="darkred",
    linestyle=":",
    linewidth=2.8,
    zorder=1,
)

# 机器平衡点文字：放在虚线右侧底部，避免与主图区域重叠
ax.text(
    RIDGE_POINT * 1.18,
    7.2e-4,
    f"机器平衡点\n{RIDGE_POINT:.1f} FLOPs/B",
    color="darkred",
    fontsize=FS_TEXT,
    ha="left",
    va="center",
)

# ==========================================
# 4. 绘制 RMSNorm 理论算术强度区间
# ==========================================
intensity_min, intensity_max = 0.67, 1.0

ax.axvspan(
    intensity_min,
    intensity_max,
    color="royalblue",
    alpha=0.14,
    zorder=0,
)

# 区域说明：放在阴影右下方，避免压住数据点
ax.text(
    1.28,
    2.4e-3,
    "RMSNorm 理论算术强度\n(Memory Bound) 区域",
    color="royalblue",
    fontsize=FS_SMALL,
    ha="left",
    va="center",
    bbox=dict(
        facecolor="white",
        edgecolor="royalblue",
        alpha=0.94,
        boxstyle="round,pad=0.45",
        linewidth=1.6,
    ),
    zorder=6,
)

# ==========================================
# 5. 绘制实际性能轨迹
# ==========================================
data_points = [
    (1, 0.67, 0.001),
    (16, 0.85, 0.05),
    (32, 0.92, 0.15),
    (64, 0.97, 0.65),
    (128, 0.99, 0.68),
]

x_data = [pt[1] for pt in data_points]
y_data = [pt[2] for pt in data_points]

ax.plot(
    x_data,
    y_data,
    color="teal",
    linestyle="--",
    linewidth=3.0,
    alpha=0.78,
    zorder=3,
)

ax.scatter(
    x_data,
    y_data,
    color="teal",
    s=130,
    edgecolor="black",
    linewidth=1.5,
    zorder=4,
    label="实测性能",
)

# B=1 标注：放在点左下方
ax.text(
    x_data[0] * 0.88,
    y_data[0] * 0.43,
    "B=1",
    color="teal",
    fontsize=FS_TEXT,
    ha="right",
    va="center",
    fontweight="bold",
)

# B>=64 标注：放在点右侧，略向下，不与绿色注释重叠
ax.text(
    1.08,
    0.50,
    r"B $\geq$ 64",
    color="teal",
    fontsize=FS_TEXT,
    ha="left",
    va="center",
    fontweight="bold",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.86,
        pad=0.2,
    ),
    zorder=7,
)

# ==========================================
# 6. 添加关键学术洞察标注
# ==========================================
theoretical_y_b1 = x_data[0] * BANDWIDTH_GBPS / 1000.0

# Latency Gap 箭头
ax.annotate(
    "",
    xy=(x_data[0], theoretical_y_b1),
    xytext=(x_data[0], y_data[0]),
    arrowprops=dict(
        arrowstyle="<|-|>",
        color="darkorange",
        lw=3.2,
    ),
    zorder=2,
)

# CPU 下发与 SM 调度开销：左移，避免与蓝色区域说明重叠
ax.text(
    0.53,
    1.35e-2,
    "CPU 下发与 SM 调度开销\n(Latency Bound)",
    color="darkorange",
    fontsize=FS_SMALL,
    ha="right",
    va="center",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.86,
        pad=0.25,
    ),
    zorder=6,
)

# 达到有效物理显存标注：上移并右移，避开 B>=64 与红线
ax.annotate(
    "达到有效物理显存\n(Bandwidth Bound)",
    xy=(x_data[-1], y_data[-1]),
    xytext=(1.95, 3.2),
    arrowprops=dict(
        arrowstyle="->",
        color="green",
        lw=2.8,
        connectionstyle="arc3,rad=.18",
    ),
    color="green",
    fontsize=FS_SMALL,
    ha="left",
    va="center",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.9,
        pad=0.28,
    ),
    zorder=7,
)

# ==========================================
# 7. 图例与布局调整
# ==========================================
legend_elements = [
    Line2D([0], [0], color="red", lw=4.2, label="A40 理论天际线"),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="teal",
        markeredgecolor="black",
        markeredgewidth=1.4,
        markersize=11,
        label="不同 Batch Size 下的实测性能",
    ),
    Line2D(
        [0],
        [0],
        color="teal",
        linestyle="--",
        lw=3,
        label="性能演变轨迹",
    ),
    Patch(
        facecolor="royalblue",
        alpha=0.14,
        label="算术强度区间 (0.67 ~ 1.0)",
    ),
]

# 图例改为两列两行，避免顶部一行过长导致挤压或裁切
ax.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.025),
    ncol=2,
    frameon=True,
    fontsize=FS_LEGEND,
    borderpad=0.55,
    labelspacing=0.65,
    handlelength=2.2,
    columnspacing=1.6,
)

# 加粗坐标轴边框
for spine in ax.spines.values():
    spine.set_linewidth(1.8)

# 布局：给顶部图例留空间，同时尽量减少四周白边
fig.subplots_adjust(
    left=0.082,
    right=0.992,
    bottom=0.105,
    top=0.80,
)

# 保存时去除多余白边，但保留极小 padding，避免文字被裁切
plt.savefig(
    "assets/图3-1.pdf",
    bbox_inches="tight",
    pad_inches=0.025,
)

# plt.show()
