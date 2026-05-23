import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# ================= 字体与全局设置 =================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 字体放大倍率：原来的 1.25 倍
FONT_SCALE = 1.7


def fs(size):
    return size * FONT_SCALE


# 略微加宽画布以容纳完整的 Warp 结构
fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

# 收紧坐标范围，减少图像内部空白
ax.set_xlim(0, 16)
ax.set_ylim(0.25, 15.25)
ax.axis("off")

# 让坐标轴铺满整个 figure，去除默认边距
ax.set_position([0, 0, 1, 1])
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# 定义配色
COLOR_WARP = "#E8F0FE"
COLOR_THREAD = "#AECBFA"
COLOR_LANE0 = "#FCE8E6"
COLOR_SMEM = "#FFE8A1"
COLOR_SYNC = "#EA4335"
COLOR_FINAL = "#34A853"
COLOR_ARROW = "#5F6368"


def draw_warp_butterfly(x_start, y_start, warp_label, is_final=False):
    """
    绘制32线程Warp的树形规约缩略图
    包含元素: T0, T1, T2, T3, ..., T16, ..., T31
    """
    # 画Warp外框
    warp_box = patches.Rectangle(
        (x_start - 0.2, y_start - 3.2),
        4.6,
        3.8,
        linewidth=1.5,
        edgecolor="#1967D2",
        facecolor=COLOR_WARP,
        linestyle="--",
    )
    ax.add_patch(warp_box)

    ax.text(
        x_start + 2.1,
        y_start + 0.3,
        warp_label,
        fontsize=fs(12),
        fontweight="bold",
        ha="center",
        color="#1967D2",
    )

    # 定义展示的视觉插槽和真实线程ID
    slots = [
        (0, "T0"),
        (1, "T1"),
        (2, "T2"),
        (3, "T3"),
        (None, "..."),
        (16, "T16"),
        (None, "..."),
        (31, "T31"),
    ]

    coords = {}
    current_y = {}

    # 1. 绘制线程块和省略号
    box_y = y_start - 0.4
    for i, (real_idx, label) in enumerate(slots):
        x = x_start + i * 0.55

        if real_idx is None:
            ax.text(
                x + 0.2,
                box_y + 0.2,
                "...",
                fontsize=fs(14),
                color="gray",
                fontweight="bold",
                ha="center",
                va="center",
            )
        else:
            color = COLOR_LANE0 if real_idx == 0 else COLOR_THREAD
            disp_label = label if not is_final else f"S{real_idx}"

            if is_final and real_idx >= 16:
                color = "#E0E0E0"
                disp_label = "0"

            rect = patches.Rectangle(
                (x, box_y),
                0.4,
                0.4,
                linewidth=1,
                edgecolor="black",
                facecolor=color,
            )
            ax.add_patch(rect)

            ax.text(
                x + 0.2,
                box_y + 0.2,
                disp_label,
                fontsize=fs(9),
                ha="center",
                va="center",
            )

            coords[real_idx] = x + 0.2
            current_y[real_idx] = box_y

    # Step 1: Gap 1
    y_gap1 = box_y - 0.4
    for src, dst in [(1, 0), (3, 2)]:
        ax.plot(
            [coords[dst], coords[dst]],
            [current_y[dst], y_gap1],
            color=COLOR_ARROW,
            lw=1.2,
        )

        ax.annotate(
            "",
            xy=(coords[dst], y_gap1),
            xytext=(coords[src], current_y[src]),
            arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=1.2),
        )

        ax.plot(coords[dst], y_gap1, "o", markersize=3, color="#1967D2")
        current_y[dst] = y_gap1

    # Step 2: Gap 2
    y_gap2 = y_gap1 - 0.4
    ax.plot(
        [coords[0], coords[0]],
        [current_y[0], y_gap2],
        color=COLOR_ARROW,
        lw=1.5,
    )

    ax.annotate(
        "",
        xy=(coords[0], y_gap2),
        xytext=(coords[2], current_y[2]),
        arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=1.5),
    )

    ax.plot(coords[0], y_gap2, "o", markersize=4, color="#1967D2")
    current_y[0] = y_gap2

    # Step 3: Gap 4 & Gap 8
    y_gap8 = y_gap2 - 0.8

    ax.plot(
        [coords[0], coords[0]],
        [current_y[0], y_gap8],
        color=COLOR_ARROW,
        lw=1.5,
        linestyle=":",
    )

    ax.plot(
        [coords[16], coords[16]],
        [current_y[16], y_gap8],
        color=COLOR_ARROW,
        lw=1.5,
        linestyle=":",
    )

    current_y[0] = y_gap8
    current_y[16] = y_gap8

    # Step 4: Gap 16
    y_gap16 = y_gap8 - 0.6

    ax.plot(
        [coords[0], coords[0]],
        [current_y[0], y_gap16],
        color=COLOR_ARROW,
        lw=2,
    )

    ax.annotate(
        "",
        xy=(coords[0], y_gap16),
        xytext=(coords[16], current_y[16]),
        arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=2),
    )

    ax.plot(coords[0], y_gap16, "s", markersize=7, color="#D93025")

    # 底部说明文字，当前保留为空
    info_text = ""
    ax.text(
        x_start + 2.1,
        y_start - 2.8,
        info_text,
        fontsize=fs(10),
        ha="center",
        color="#1967D2",
    )

    return coords[0], y_gap16


# ----------------- 绘制第一阶段 -----------------
ax.text(
    0.5,
    14.8,
    "第一阶段：Warp级内部无锁树形规约",
    fontsize=fs(14),
    fontweight="bold",
)

res_w0_x, res_w0_y = draw_warp_butterfly(0.5, 13.5, "Warp 0 (Active)")
res_w1_x, res_w1_y = draw_warp_butterfly(5.5, 13.5, "Warp 1 (Active)")

ax.text(
    10.1,
    12,
    "...",
    fontsize=fs(24),
    color="gray",
    fontweight="bold",
)

res_wn_x, res_wn_y = draw_warp_butterfly(11.0, 13.5, "Warp N (Active)")


# ----------------- 绘制第二阶段 -----------------
ax.text(
    0.5,
    9.5,
    "第二阶段：轻量级Shared Memory暂存与全局同步",
    fontsize=fs(14),
    fontweight="bold",
)

# Shared Memory Block
smem_y = 7.5
smem_width = 8

smem_box = patches.Rectangle(
    (4, smem_y),
    smem_width,
    1,
    linewidth=2,
    edgecolor="#E37400",
    facecolor=COLOR_SMEM,
)
ax.add_patch(smem_box)

ax.text(
    8,
    smem_y + 0.5,
    "Shared Memory Array (暂存各Warp局部累加和)",
    fontsize=fs(12),
    fontweight="bold",
    ha="center",
    va="center",
    color="#B06000",
)

# 写入箭头
for x_start_arrow, res_x, res_y in [
    (4.5, res_w0_x, res_w0_y),
    (6.5, res_w1_x, res_w1_y),
    (11.5, res_wn_x, res_wn_y),
]:
    ax.annotate(
        "",
        xy=(x_start_arrow, smem_y + 1),
        xytext=(res_x, res_y),
        arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=1.5, ls="--"),
    )

ax.text(
    3,
    8.8,
    "Leader线程(Lane 0)写入落盘",
    fontsize=fs(10),
    color="#E37400",
)

# 同步屏障
sync_y = 6.8
ax.plot([0, 16], [sync_y, sync_y], color=COLOR_SYNC, lw=4, linestyle="-.")

text = ax.text(
    2.4,
    sync_y + 0.15,
    "强制全局同步屏障: __syncthreads()",
    fontsize=fs(14),
    fontweight="bold",
    color=COLOR_SYNC,
    ha="center",
)
text.set_path_effects([path_effects.withStroke(linewidth=3, foreground="white")])


# ----------------- 绘制第三阶段 -----------------
third_stage_title_y = 6
third_stage_warp_y = 5

ax.text(
    0.5,
    third_stage_title_y,
    "第三阶段：单一Warp唤醒规约",
    fontsize=fs(14),
    fontweight="bold",
)

res_final_x, res_final_y = draw_warp_butterfly(
    5.5,
    third_stage_warp_y,
    "仅 Warp 0 活跃",
    is_final=True,
)

# Zero-padding 标注
ax.annotate(
    "越界元素进行\nZero-padding",
    xy=(9.75, third_stage_warp_y),
    xytext=(11.5, third_stage_warp_y),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=.2",
        color="#808080",
    ),
    fontsize=fs(10),
    color="#808080",
)

# 读取箭头
for x_end_arrow, x_start_arrow in [(5.7, 4.5), (6.2, 6.5), (8, 11.5)]:
    ax.annotate(
        "",
        xy=(x_end_arrow, third_stage_warp_y),
        xytext=(x_start_arrow, smem_y),
        arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=1.5),
    )

# 最终结果
final_box = patches.Rectangle(
    (res_final_x - 1.5, res_final_y - 2),
    3,
    1,
    linewidth=2,
    edgecolor="#0D652D",
    facecolor=COLOR_FINAL,
)
ax.add_patch(final_box)

ax.text(
    res_final_x,
    res_final_y - 1.5,
    "最终规约结果",
    fontsize=fs(11),
    fontweight="bold",
    color="black",
    ha="center",
    va="center",
)

# 最终箭头
ax.annotate(
    "",
    xy=(res_final_x, res_final_y - 1),
    xytext=(res_final_x, res_final_y),
    arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=2),
)

# 去除坐标轴内部留白
ax.margins(0)

# 保存时去除上下左右白边
plt.savefig(
    "assets/图4-2.pdf",
    bbox_inches="tight",
    pad_inches=0,
)

plt.show()
