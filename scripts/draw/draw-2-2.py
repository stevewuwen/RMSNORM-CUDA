import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= 字体与全局设置 =================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

# 扩展画布边界，防止阴影被边缘裁切
fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300)
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-0.5, 9.5)
ax.axis("off")

# ================= 现代扁平化 2.0 颜色体系 =================
COLOR_SHADOW = "#0F172A"
ALPHA_SHADOW = 0.08

# 内存节点 (青蓝色系)
COLOR_MEM_BG = "#E0F2FE"
COLOR_MEM_BORDER = "#0284C7"

# 计算节点 (黑白高对比)
COLOR_NODE_BG = "#FFFFFF"
COLOR_NODE_BORDER = "#334155"

# 模块容器
COLOR_PASS1_BG = "#F0F9FF"
COLOR_PASS1_BORDER = "#7DD3FC"
COLOR_PASS2_BG = "#FFF7ED"
COLOR_PASS2_BORDER = "#FDBA74"

# 参数节点
COLOR_PARAM_BG = "#F3E8FF"
COLOR_PARAM_BORDER = "#9333EA"

# 数据流与高亮
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
    w=1.6,
    h=1.2,  # 【优化】为了适应大字体，默认节点尺寸放大
    facecolor=COLOR_NODE_BG,
    edgecolor=COLOR_NODE_BORDER,
    fontsize=24,  # 【优化】默认公式字体从 18 调大到 24
    text_color="#0F172A",
):
    boxstyle = "round,pad=0.05,rounding_size=0.2"
    draw_shadow_box(ax, x, y, w, h, boxstyle, zorder=4)
    box = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.5,
        zorder=5,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
        zorder=6,
    )


def draw_container(ax, x, y, w, h, title, title_color, facecolor, edgecolor):
    boxstyle = "round,pad=0.2,rounding_size=0.3"
    draw_shadow_box(ax, x, y, w, h, boxstyle, zorder=0, offset=(0.15, -0.15))
    box = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle="--",
        linewidth=2.5,
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(
        x - w / 2 + 0.3,
        y + h / 2 - 0.45,  # 【优化】调整标题的 Y 轴偏移量
        title,
        fontsize=18,  # 【优化】容器标题从 13 调大到 18
        color=title_color,
        fontweight="bold",
        zorder=2,
    )


def draw_ortho_arrow(
    ax, x1, y1, x2, y2, x_mid=None, color=COLOR_FLOW, lw=3, label=None, label_coord=None
):
    if x_mid is None:
        x_mid = x1 + (x2 - x1) * 0.5
    ax.plot(
        [x1, x_mid, x_mid, x2 - 0.15], [y1, y1, y2, y2], color=color, lw=lw, zorder=2
    )
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x2 - 0.15, y2),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=22),
        zorder=3,
    )
    if label and label_coord:
        bbox_props = dict(
            boxstyle="round,pad=0.3,rounding_size=0.2",
            facecolor=color,
            edgecolor="none",
        )
        ax.text(
            label_coord[0],
            label_coord[1],
            label,
            color="white",
            fontsize=16,  # 【优化】线条文字标签从 11 调大到 16
            fontweight="bold",
            ha="center",
            va="center",
            bbox=bbox_props,
            zorder=6,
        )


def draw_straight_arrow(ax, x1, y1, x2, y2, color=COLOR_FLOW, lw=3):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=22),
        zorder=2,
    )


# ================= 1. 绘制背景区域容器 =================
# 【优化】为了包裹住变大的节点，容器宽高度(W, H)均有小幅增加
draw_container(
    ax,
    7.3,
    6.7,
    8.6,
    2.4,
    "第一次遍历: 聚合统计量",
    "#0284C7",
    COLOR_PASS1_BG,
    COLOR_PASS1_BORDER,
)
draw_container(
    ax,
    8.2,
    2.5,
    9.8,
    2.4,
    "第二次遍历: 逐元素缩放",
    "#C2410C",
    COLOR_PASS2_BG,
    COLOR_PASS2_BORDER,
)


# ================= 2. 放置全部节点 =================
# 【优化】输入输出节点变大，字体调到 32
draw_node(
    ax,
    1.5,
    4.5,
    r"$\mathbf{X}$",
    w=1.4,
    h=1.4,
    facecolor=COLOR_MEM_BG,
    edgecolor=COLOR_MEM_BORDER,
    fontsize=32,
    text_color="#0369A1",
)
draw_node(
    ax,
    14.2,
    2.5,
    r"$\mathbf{Y}$",
    w=1.4,
    h=1.4,
    facecolor=COLOR_MEM_BG,
    edgecolor=COLOR_MEM_BORDER,
    fontsize=32,
    text_color="#0369A1",
)

# Pass 1 节点
draw_node(ax, 4.5, 6.5, r"$(\cdot)^2$")
draw_node(ax, 7.0, 6.5, r"$\frac{1}{H}\sum$")
# 【优化】公式长，为了容纳 24 号字体，宽度从 2.1 增加到 2.8
draw_node(ax, 10.0, 6.5, r"$1/\sqrt{\cdot + \epsilon}$", w=2.8)

# Pass 2 节点 (乘法符号调至 28)
draw_node(ax, 10.0, 2.5, r"$\otimes$", w=1.2, h=1.2, fontsize=28)
draw_node(ax, 12.0, 2.5, r"$\otimes$", w=1.2, h=1.2, fontsize=28)

# 参数 Gamma
draw_node(
    ax,
    12.0,
    0.3,
    r"$\gamma$",
    w=1.0,
    h=1.0,
    facecolor=COLOR_PARAM_BG,
    edgecolor=COLOR_PARAM_BORDER,
    fontsize=28,
    text_color="#7E22CE",
)


# ================= 3. 绘制精确连线 =================
# 【优化】由于节点变大，对所有箭头的起止坐标做了精细重算，防止箭头插进节点里
draw_ortho_arrow(
    ax,
    2.3,
    4.5,
    3.6,
    6.5,
    x_mid=2.9,
    color=COLOR_READ,
    label="Read 1",
    label_coord=(2.9, 5.5),
)
draw_ortho_arrow(
    ax,
    2.3,
    4.5,
    9.3,
    2.5,
    x_mid=2.9,
    color=COLOR_READ,
    label="Read 2",
    label_coord=(6.0, 2.5),
)

# Pass 1 内部连线
draw_straight_arrow(ax, 5.4, 6.5, 6.1, 6.5)
draw_straight_arrow(ax, 7.9, 6.5, 8.5, 6.5)

# Pass 1 到 Pass 2
ax.plot(
    [10.0, 10.0], [5.8, 3.2], color=COLOR_FLOW, lw=2.5, linestyle=(0, (5, 3)), zorder=2
)
ax.annotate(
    "",
    xy=(10.0, 3.1),
    xytext=(10.0, 3.2),
    arrowprops=dict(arrowstyle="-|>", color=COLOR_FLOW, lw=2.5, mutation_scale=22),
    zorder=3,
)

# 标量标签
bbox_scalar = dict(
    boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.5
)
ax.text(
    10.0,
    4.5,
    r"标量 $r$",
    fontsize=18,  # 【优化】标量文字从 13 调大到 18
    color="#334155",
    fontweight="bold",
    ha="center",
    va="center",
    bbox=bbox_scalar,
    zorder=4,
)

# Pass 2 内部连线
draw_straight_arrow(ax, 10.7, 2.5, 11.3, 2.5)
draw_straight_arrow(ax, 12.0, 1, 12.0, 1.8)
draw_straight_arrow(ax, 12.7, 2.5, 13.4, 2.5)


# ================= 4. 标题与收尾 =================
plt.text(
    7.5,
    9.0,
    "图 2-2 原生 RMSNorm 计算流程图",
    fontsize=26,  # 【优化】主标题从 20 调大到 26
    fontweight="900",
    color="#1E293B",
    ha="center",
)

plt.tight_layout()
plt.savefig("assets/图2-2.pdf", bbox_inches="tight")
