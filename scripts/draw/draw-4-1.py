import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体，防止中文显示乱码
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15, 9), dpi=150)
ax.set_xlim(-22, 172)
ax.set_ylim(-2, 112)
ax.axis("off")  # 隐藏原始坐标轴

# 访存数据块颜色
col_scalar = "#FF6B6B"  # 红色（低效）
col_pack64 = "#4D96FF"  # 蓝色（均衡）
col_pack128 = "#20BF55"  # 绿色（满载）

# 状态与背景
col_bg = "#F3F4F7"
col_empty = "#E0E0E0"
col_text_main = "#2B2D42"
col_text_sub = "#6C757D"

# 寄存器压力颜色梯度
col_reg_low = "#88D49E"  # 安全
col_reg_mid = "#FFD166"  # 适中
col_reg_high = "#EF476F"  # 危险

# ==================== 顶部自解释图例 ====================
# 访存总线图例
ax.add_patch(
    patches.Rectangle(
        (0, 102),
        10,
        4,
        facecolor=col_empty,
        edgecolor="#888",
        linestyle="--",
        lw=1.5,
    )
)
ax.text(12, 104, "128-bit总线", va="center", fontsize=11)

ax.add_patch(
    patches.Rectangle((58, 102), 6, 4, facecolor="#4D96FF", edgecolor="white", lw=1)
)
ax.text(66, 104, "有效 FP16 数据", va="center", fontsize=11)

# 寄存器压力图例
ax.text(102, 104, "寄存器占用:", va="center", fontsize=11)
colors = [col_reg_low, col_reg_mid, col_reg_high]
labels = ["低", "中", "高"]
for i in range(3):
    ax.add_patch(
        patches.Rectangle(
            (128 + i * 11, 102), 4, 4, facecolor=colors[i], edgecolor="none"
        )
    )
    ax.text(133 + i * 11, 104, labels[i], va="center", fontsize=10)


# ==================== 绘制行函数 ====================
def draw_row(y, name, align, fp16_count, reg_level, util_str, req_str):
    # 总线基准坐标
    x_start = 12
    bus_w = 112  # 112对应128-bit物理总线宽度 (1单位 = 1.14 bit)
    fp16_w = 14  # 每个 FP16 占 16-bit，物理宽度为14

    # 1. 绘制 128-bit 总线背景
    ax.add_patch(
        patches.Rectangle(
            (x_start, y + 8),
            bus_w,
            8,
            facecolor=col_bg,
            edgecolor="#A0A0A0",
            linestyle="--",
            lw=1.5,
            zorder=1,
        )
    )

    # 2. 绘制有效 FP16 数据块
    color_map = {1: col_scalar, 4: col_pack64, 8: col_pack128}
    current_color = color_map[fp16_count]

    for i in range(fp16_count):
        rect = patches.Rectangle(
            (x_start + i * fp16_w, y + 8),
            fp16_w,
            8,
            facecolor=current_color,
            edgecolor="white",
            lw=1.5,
            zorder=2,
        )
        ax.add_patch(rect)
        # 在数据块内印上“F16”
        ax.text(
            x_start + i * fp16_w + fp16_w / 2,
            y + 12,
            "FP16",
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )

    # 绘制多余的空载区域指示
    if fp16_count < 8:
        empty_w = bus_w - (fp16_count * fp16_w)
        ax.add_patch(
            patches.Rectangle(
                (x_start + fp16_count * fp16_w, y + 8),
                empty_w,
                8,
                facecolor=col_empty,
                edgecolor="none",
                alpha=0.4,
                zorder=1,
            )
        )

    # 3. 新增：寄存器压力条（下半部分）
    reg_w_map = {1: 14, 2: 56, 3: 112}  # 压力宽度映射
    reg_col_map = {1: col_reg_low, 2: col_reg_mid, 3: col_reg_high}

    # 背景槽
    ax.add_patch(
        patches.Rectangle(
            (x_start, y + 1),
            bus_w,
            3,
            facecolor="#E9ECEF",
            edgecolor="none",
            zorder=1,
        )
    )
    # 实际占用
    ax.add_patch(
        patches.Rectangle(
            (x_start, y + 1),
            reg_w_map[reg_level],
            3,
            facecolor=reg_col_map[reg_level],
            edgecolor="none",
            zorder=2,
        )
    )

    # 4. 左侧描述区域
    # 对齐指示线
    align_colors = {"2字节": "#FFAAA6", "8字节": "#95D1CC", "16字节": "#A2D2FF"}
    ax.plot(
        [x_start - 1, x_start - 1],
        [y, y + 17],
        color=align_colors[align],
        lw=4,
        solid_capstyle="round",
    )

    ax.text(
        x_start - 4,
        y + 12,
        name,
        ha="right",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=col_text_main,
    )


# ==================== 绘制三层对比 ====================
# Pack128 (最上层)
draw_row(
    y=67,
    name="向量化 Pack128",
    align="16字节",
    fp16_count=8,
    reg_level=3,
    util_str="100.0% (满载)",
    req_str="12.5% (最低)",
)

# Pack64 (中间层)
draw_row(
    y=36,
    name="向量化 Pack64",
    align="8字节",
    fp16_count=4,
    reg_level=2,
    util_str="50.0% (适中)",
    req_str="25.0% (较低)",
)

# Scalar (最底层)
draw_row(
    y=5,
    name="传统标量访存",
    align="2字节",
    fp16_count=1,
    reg_level=1,
    util_str="12.5% (极低)",
    req_str="100.0% (基准)",
)

# ==================== 标题区域 ====================
plt.text(
    75,
    113,
    "图 4-1  16-bit 标量访存与 64/128-bit 并发向量化访存综合对比图",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
)

# 调整布局并保存
plt.tight_layout()
plt.savefig("assets/图4-1.pdf", bbox_inches="tight")
