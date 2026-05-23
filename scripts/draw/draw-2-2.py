import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= 字体与全局设置 =================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

# 字体缩放倍率：原来为 1，这里稍微调小
FONT_SCALE = 0.88


def scaled_font(size):
    return int(size * FONT_SCALE)


fig, ax = plt.subplots(figsize=(18, 8.5), dpi=300)

# 收紧画布范围，减少坐标系本身带来的空白
ax.set_xlim(0.0, 20.0)
ax.set_ylim(-0.8, 8.8)
ax.axis("off")

# 让坐标轴占满整个 figure，避免 matplotlib 默认边距
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# ================= 现代扁平化 2.0 颜色体系 =================
COLOR_SHADOW = "#0F172A"
ALPHA_SHADOW = 0.08

COLOR_MEM_BG = "#E0F2FE"
COLOR_MEM_BORDER = "#0284C7"

COLOR_NODE_BG = "#FFFFFF"
COLOR_NODE_BORDER = "#334155"

COLOR_PASS1_BG = "#F0F9FF"
COLOR_PASS1_BORDER = "#7DD3FC"
COLOR_PASS2_BG = "#FFF7ED"
COLOR_PASS2_BORDER = "#FDBA74"

COLOR_PARAM_BG = "#F3E8FF"
COLOR_PARAM_BORDER = "#9333EA"

COLOR_READ = "#EA580C"
COLOR_FLOW = "#64748B"


# ================= 绘制辅助函数 =================
def draw_shadow_box(ax, x, y, w, h, boxstyle, zorder, offset=(0.12, -0.12)):
    shadow = patches.FancyBboxPatch(
        (x - w / 2 + offset[0], y - h / 2 + offset[1]),
        w,
        h,
        boxstyle=boxstyle,
        facecolor=COLOR_SHADOW,
        edgecolor="none",
        alpha=ALPHA_SHADOW,
        zorder=zorder,
    )
    ax.add_patch(shadow)


def draw_node(
    ax,
    x,
    y,
    text,
    w=2.4,
    h=1.5,
    facecolor=COLOR_NODE_BG,
    edgecolor=COLOR_NODE_BORDER,
    fontsize=48,
    text_color="#0F172A",
):
    boxstyle = "round,pad=0.08,rounding_size=0.24"

    draw_shadow_box(ax, x, y, w, h, boxstyle, zorder=4)

    box = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=3.2,
        zorder=5,
    )
    ax.add_patch(box)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=scaled_font(fontsize),
        color=text_color,
        fontweight="bold",
        zorder=6,
    )


def draw_container(ax, x, y, w, h, title, title_color, facecolor, edgecolor):
    boxstyle = "round,pad=0.25,rounding_size=0.36"

    draw_shadow_box(ax, x, y, w, h, boxstyle, zorder=0, offset=(0.15, -0.15))

    box = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle="--",
        linewidth=3.2,
        zorder=1,
    )
    ax.add_patch(box)

    ax.text(
        x - w / 2 + 0.35,
        y + h / 2 - 0.62,
        title,
        fontsize=scaled_font(36),
        color=title_color,
        fontweight="bold",
        zorder=2,
    )


def draw_ortho_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    x_mid=None,
    color=COLOR_FLOW,
    lw=4,
    label=None,
    label_coord=None,
):
    if x_mid is None:
        x_mid = x1 + (x2 - x1) * 0.5

    ax.plot(
        [x1, x_mid, x_mid, x2 - 0.18],
        [y1, y1, y2, y2],
        color=color,
        lw=lw,
        zorder=2,
    )

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x2 - 0.18, y2),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=34,
        ),
        zorder=3,
    )

    if label and label_coord:
        bbox_props = dict(
            boxstyle="round,pad=0.32,rounding_size=0.22",
            facecolor=color,
            edgecolor="none",
        )

        ax.text(
            label_coord[0],
            label_coord[1],
            label,
            color="white",
            fontsize=scaled_font(32),
            fontweight="bold",
            ha="center",
            va="center",
            bbox=bbox_props,
            zorder=6,
        )


def draw_straight_arrow(ax, x1, y1, x2, y2, color=COLOR_FLOW, lw=4):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=34,
        ),
        zorder=2,
    )


# ================= 1. 绘制背景区域容器 =================
draw_container(
    ax,
    9.6,
    6.6,
    11.8,
    2.9,
    "第一次遍历: 聚合统计量",
    "#0284C7",
    COLOR_PASS1_BG,
    COLOR_PASS1_BORDER,
)

draw_container(
    ax,
    10.7,
    2.4,
    13.8,
    2.9,
    "第二次遍历: 逐元素缩放",
    "#C2410C",
    COLOR_PASS2_BG,
    COLOR_PASS2_BORDER,
)


# ================= 2. 放置全部节点 =================
draw_node(
    ax,
    1.5,
    4.5,
    r"$\mathbf{X}$",
    w=1.8,
    h=1.6,
    facecolor=COLOR_MEM_BG,
    edgecolor=COLOR_MEM_BORDER,
    fontsize=64,
    text_color="#0369A1",
)

draw_node(
    ax,
    19,
    2.4,
    r"$\mathbf{Y}$",
    w=1.8,
    h=1.6,
    facecolor=COLOR_MEM_BG,
    edgecolor=COLOR_MEM_BORDER,
    fontsize=64,
    text_color="#0369A1",
)

# Pass 1 节点
draw_node(ax, 5.0, 6.4, r"$(\cdot)^2$", w=2.4, h=1.5, fontsize=48)
draw_node(ax, 8.2, 6.4, r"$\frac{1}{H}\sum$", w=2.8, h=1.5, fontsize=48)
draw_node(ax, 12.4, 6.4, r"$1/\sqrt{\cdot + \epsilon}$", w=4.4, h=1.5, fontsize=48)

# Pass 2 节点
draw_node(ax, 12.4, 2.4, r"$\otimes$", w=1.6, h=1.4, fontsize=56)
draw_node(ax, 15.3, 2.4, r"$\otimes$", w=1.6, h=1.4, fontsize=56)

# 参数 Gamma
draw_node(
    ax,
    15.3,
    0.15,
    r"$\gamma$",
    w=1.4,
    h=1.25,
    facecolor=COLOR_PARAM_BG,
    edgecolor=COLOR_PARAM_BORDER,
    fontsize=56,
    text_color="#7E22CE",
)


# ================= 3. 绘制连线 =================
draw_ortho_arrow(
    ax,
    2.45,
    4.5,
    3.75,
    6.4,
    x_mid=3.05,
    color=COLOR_READ,
    label="Read 1",
    label_coord=(3.2, 5.45),
)

draw_ortho_arrow(
    ax,
    2.45,
    4.5,
    11.45,
    2.4,
    x_mid=3.05,
    color=COLOR_READ,
    label="Read 2",
    label_coord=(6.8, 2.4),
)

# Pass 1 内部连线
draw_straight_arrow(ax, 6.25, 6.4, 6.8, 6.4)
draw_straight_arrow(ax, 9, 6.4, 10.2, 6.4)

# Pass 1 到 Pass 2
ax.plot(
    [12.4, 12.4],
    [5.6, 3.15],
    color=COLOR_FLOW,
    lw=3.6,
    linestyle=(0, (5, 3)),
    zorder=2,
)

ax.annotate(
    "",
    xy=(12.4, 3.05),
    xytext=(12.4, 3.18),
    arrowprops=dict(
        arrowstyle="-|>",
        color=COLOR_FLOW,
        lw=3.6,
        mutation_scale=34,
    ),
    zorder=3,
)

# 标量标签
bbox_scalar = dict(
    boxstyle="round,pad=0.25",
    facecolor="#FFFFFF",
    edgecolor="#CBD5E1",
    lw=1.8,
)

ax.text(
    12.4,
    4.45,
    r"标量 $r$",
    fontsize=scaled_font(36),
    color="#334155",
    fontweight="bold",
    ha="center",
    va="center",
    bbox=bbox_scalar,
    zorder=4,
)

# Pass 2 内部连线
draw_straight_arrow(ax, 13.25, 2.4, 14.45, 2.4)
draw_straight_arrow(ax, 15.3, 0.85, 15.3, 1.6)
draw_straight_arrow(ax, 16.15, 2.4, 18, 2.4)


# ================= 4. 保存 =================
plt.savefig(
    "assets/图2-2.pdf",
    bbox_inches="tight",
    pad_inches=0,
    facecolor="white",
)
plt.close()
