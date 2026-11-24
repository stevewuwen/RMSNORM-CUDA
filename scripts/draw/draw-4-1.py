import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# ==================== 字体与全局样式 ====================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 字体放大倍数
FONT_SCALE = 3

# 原始字体大小 × 3
FS_LEGEND = 11 * FONT_SCALE
FS_LEGEND_SMALL = 10 * FONT_SCALE
FS_ROW_NAME = 14 * FONT_SCALE
FS_BLOCK = 9 * FONT_SCALE
FS_RIGHT = 11 * FONT_SCALE

# ==================== 画布设置 ====================
fig, ax = plt.subplots(figsize=(28, 16), dpi=200)

ax.set_xlim(-90, 680)
ax.set_ylim(-10, 390)
ax.axis("off")

# ==================== 颜色设置 ====================
col_scalar = "#FF6B6B"  # 标量访存
col_pack64 = "#4D96FF"  # Pack64
col_pack128 = "#20BF55"  # Pack128

col_bg = "#F3F4F7"
col_empty = "#E0E0E0"
col_text_main = "#2B2D42"
col_text_sub = "#6C757D"

col_reg_low = "#88D49E"
col_reg_mid = "#FFD166"
col_reg_high = "#EF476F"

# ==================== 顶部图例 ====================
legend_y = 350

# 128-bit 总线图例
ax.add_patch(
    patches.Rectangle(
        (40, legend_y),
        42,
        16,
        facecolor=col_empty,
        edgecolor="#888888",
        linestyle="--",
        lw=2.5,
    )
)
ax.text(
    95,
    legend_y + 8,
    "128-bit 总线",
    va="center",
    fontsize=FS_LEGEND,
    color=col_text_main,
)

# 有效 FP16 数据图例
ax.add_patch(
    patches.Rectangle(
        (250, legend_y),
        42,
        16,
        facecolor=col_pack64,
        edgecolor="white",
        lw=2,
    )
)
ax.text(
    305,
    legend_y + 8,
    "有效 FP16 数据",
    va="center",
    fontsize=FS_LEGEND,
    color=col_text_main,
)

# 寄存器压力图例
ax.text(
    475,
    legend_y + 8,
    "寄存器占用:",
    va="center",
    fontsize=FS_LEGEND,
    color=col_text_main,
)

colors = [col_reg_low, col_reg_mid, col_reg_high]
labels = ["低", "中", "高"]

for i in range(3):
    ax.add_patch(
        patches.Rectangle(
            (575 + i * 35, legend_y),
            14,
            16,
            facecolor=colors[i],
            edgecolor="none",
        )
    )
    ax.text(
        594 + i * 35,
        legend_y + 8,
        labels[i],
        va="center",
        fontsize=FS_LEGEND_SMALL,
        color=col_text_main,
    )


# ==================== 绘制单行函数 ====================
def draw_row(y, name, align, fp16_count, reg_level, util_str, req_str):
    x_start = 150

    # 128-bit 总线宽度：8 个 FP16 × 每块 45
    fp16_w = 45
    bus_w = fp16_w * 8

    bus_h = 30
    reg_h = 10

    # 1. 总线背景
    ax.add_patch(
        patches.Rectangle(
            (x_start, y + 34),
            bus_w,
            bus_h,
            facecolor=col_bg,
            edgecolor="#A0A0A0",
            linestyle="--",
            lw=2.5,
            zorder=1,
        )
    )

    # 2. 有效 FP16 数据块
    color_map = {
        1: col_scalar,
        4: col_pack64,
        8: col_pack128,
    }
    current_color = color_map[fp16_count]

    for i in range(fp16_count):
        rect = patches.Rectangle(
            (x_start + i * fp16_w, y + 34),
            fp16_w,
            bus_h,
            facecolor=current_color,
            edgecolor="white",
            lw=2.5,
            zorder=2,
        )
        ax.add_patch(rect)

        ax.text(
            x_start + i * fp16_w + fp16_w / 2,
            y + 34 + bus_h / 2,
            "FP16",
            color="white",
            ha="center",
            va="center",
            fontsize=FS_BLOCK,
            fontweight="bold",
            zorder=3,
        )

    # 3. 空载区域
    if fp16_count < 8:
        empty_w = bus_w - fp16_count * fp16_w
        ax.add_patch(
            patches.Rectangle(
                (x_start + fp16_count * fp16_w, y + 34),
                empty_w,
                bus_h,
                facecolor=col_empty,
                edgecolor="none",
                alpha=0.45,
                zorder=1,
            )
        )

    # 4. 寄存器压力条
    reg_w_map = {
        1: bus_w * 0.125,
        2: bus_w * 0.5,
        3: bus_w,
    }
    reg_col_map = {
        1: col_reg_low,
        2: col_reg_mid,
        3: col_reg_high,
    }

    ax.add_patch(
        patches.Rectangle(
            (x_start, y + 12),
            bus_w,
            reg_h,
            facecolor="#E9ECEF",
            edgecolor="none",
            zorder=1,
        )
    )

    ax.add_patch(
        patches.Rectangle(
            (x_start, y + 12),
            reg_w_map[reg_level],
            reg_h,
            facecolor=reg_col_map[reg_level],
            edgecolor="none",
            zorder=2,
        )
    )

    # 5. 左侧对齐标识线
    align_colors = {
        "2字节": "#FFAAA6",
        "8字节": "#95D1CC",
        "16字节": "#A2D2FF",
    }

    ax.plot(
        [x_start - 18, x_start - 18],
        [y + 4, y + 74],
        color=align_colors[align],
        lw=8,
        solid_capstyle="round",
    )

    # 6. 左侧名称
    ax.text(
        x_start - 32,
        y + 49,
        name,
        ha="right",
        va="center",
        fontsize=FS_ROW_NAME,
        fontweight="bold",
        color=col_text_main,
    )

    # 7. 右侧指标说明
    right_x = x_start + bus_w + 35


# ==================== 绘制三层对比 ====================
draw_row(
    y=250,
    name="向量化 Pack128",
    align="16字节",
    fp16_count=8,
    reg_level=3,
    util_str="100.0%（满载）",
    req_str="12.5%（最低）",
)

draw_row(
    y=145,
    name="向量化 Pack64",
    align="8字节",
    fp16_count=4,
    reg_level=2,
    util_str="50.0%（适中）",
    req_str="25.0%（较低）",
)

draw_row(
    y=40,
    name="传统标量访存",
    align="2字节",
    fp16_count=1,
    reg_level=1,
    util_str="12.5%（极低）",
    req_str="100.0%（基准）",
)

# ==================== 保存 ====================
Path("assets").mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig("assets/图4-1.pdf", bbox_inches="tight")
