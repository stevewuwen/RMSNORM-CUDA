import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==================== 1. 初始化与样式配置 ====================
# 设置全局字体和背景色
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
fig, ax = plt.subplots(figsize=(16, 6.5), dpi=300)
ax.set_facecolor("#F8F9FA")
ax.set_xlim(0, 24)
ax.set_ylim(-1.5, 8.5)
ax.axis("off")

# 现代感配色
C_PREFILL = "#00C4B6"  # 青色：代表高并发的 Prefill
C_DECODE = "#F39C12"  # 橙色：代表串行的 Decode
C_CACHE = "#3498DB"  # 蓝色：代表 K/V Cache
C_TOKEN = "#E74C3C"  # 红色：代表生成的 Token 边框
C_EDGE = "#2C3E50"  # 深灰蓝：边框和文字
C_SHADOW = "black"  # 阴影颜色


# ==================== 2. 核心绘图辅助函数 ====================
def draw_box(
    x,
    y,
    w,
    h,
    text="",
    facecolor="white",
    text_color=C_EDGE,
    zorder=3,
    shadow_offset=0.1,
):
    """绘制现代圆角矩形（带阴影特效）"""
    shadow = patches.FancyBboxPatch(
        (x + shadow_offset, y - shadow_offset),
        w,
        h,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        linewidth=0,
        facecolor=C_SHADOW,
        alpha=0.15,
        zorder=zorder - 1,
    )
    ax.add_patch(shadow)

    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        linewidth=2,
        edgecolor=C_EDGE,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(box)

    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            fontstyle="italic" if "User" in text else "normal",
            color=text_color,
            zorder=4,
        )


def draw_arrow(x_start, y_start, x_end, y_end, lw=2.5, rad=0.0):
    """绘制防重叠水平/垂直/斜向箭头，支持带弧度"""
    ax.annotate(
        "",
        xy=(x_end, y_end),
        xycoords="data",
        xytext=(x_start, y_start),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="->,head_width=0.35,head_length=0.45",
            linewidth=lw,
            color=C_EDGE,
            shrinkA=3,
            shrinkB=3,  # 缩进：防止箭头戳破边框
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=2,
    )


def draw_cache_blocks(center_x, y, count, is_new=False):
    """绘制底部的 K/V Cache 小方块，自动居中对齐上方计算节点"""
    block_w, block_h, gap = 0.6, 0.6, 0
    total_w = count * block_w + (count - 1) * gap
    start_x = center_x - total_w / 2

    for i in range(count):
        is_last = i == count - 1
        lw = 2.5 if (is_new and is_last) else 1.5
        ec = C_TOKEN if (is_new and is_last) else C_EDGE
        fc = C_CACHE

        rect = patches.Rectangle(
            (start_x + i * (block_w + gap), y),
            block_w,
            block_h,
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
        )
        ax.add_patch(rect)


def get_cache_block_center(center_x, count, target_idx):
    """【新增】获取指定 Cache 小方块的中心 X 坐标，用于箭头精准制导"""
    block_w, gap = 0.6, 0.2
    total_w = count * block_w + (count - 1) * gap
    start_x = center_x - total_w / 2
    return start_x + target_idx * (block_w + gap) + block_w / 2


# ==================== 3. 绘制文字与输入区 ====================
ax.text(
    1.2,
    4.5,
    "大语言模型\n推理概述",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    fontstyle="italic",
    color=C_EDGE,
)

draw_box(2.8, 3.8, 1.8, 1.2, "用户提示词", facecolor="#ECF0F1")
draw_arrow(4.7, 4.4, 5.5, 4.4)

# ==================== 4. 计算层：Prefill 与 Decode (流水线形态) ====================
Y_COMPUTE = 3.6
H_COMPUTE = 1.6
Y_ARROW = 4.4
Y_TOKEN = 4.65

# --- Prefill ---
draw_box(5.5, Y_COMPUTE, 4.0, H_COMPUTE, "Prefill", facecolor=C_PREFILL)
draw_arrow(9.5, Y_ARROW, 12.5, Y_ARROW)
draw_box(10.6, Y_TOKEN, 0.8, 0.7, "$T_1$", facecolor=C_PREFILL, shadow_offset=0.05)

# --- Decode 1 ---
draw_box(12.5, Y_COMPUTE, 2.6, H_COMPUTE, "Decode", facecolor=C_DECODE)
draw_arrow(14.9, Y_ARROW, 17.8, Y_ARROW)
draw_box(15.95, Y_TOKEN, 0.8, 0.7, "$T_2$", facecolor=C_DECODE, shadow_offset=0.05)

# --- Decode 2 ---
draw_box(17.8, Y_COMPUTE, 2.6, H_COMPUTE, "Decode", facecolor=C_DECODE)
draw_arrow(20.2, Y_ARROW, 22.5, Y_ARROW)
draw_box(20.95, Y_TOKEN, 0.8, 0.7, "$T_3$", facecolor=C_DECODE, shadow_offset=0.05)

ax.text(23.0, Y_ARROW, "...", fontsize=24, fontweight="bold", color=C_EDGE)
ax.text(
    16.35,
    6.0,
    "自回归地逐个生成token",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    fontstyle="italic",
    color=C_EDGE,
)

# ==================== 5. 显存层：K/V Cache 读写逻辑优化 ====================
Y_CACHE_TRACK = 0.5

# 显存池背景装饰轨道
track_rect = patches.Rectangle(
    (5.3, Y_CACHE_TRACK - 0.2), 17.0, 1.0, linewidth=0, facecolor="#EBF5FB", zorder=1
)
ax.add_patch(track_rect)
ax.plot(
    [5.3, 22.3],
    [Y_CACHE_TRACK + 1.0, Y_CACHE_TRACK + 1.0],
    color=C_EDGE,
    linewidth=1.5,
    linestyle="--",
    zorder=2,
)
ax.plot(
    [5.3, 22.3],
    [Y_CACHE_TRACK - 0.2, Y_CACHE_TRACK - 0.2],
    color=C_EDGE,
    linewidth=2,
    zorder=2,
)

ax.text(
    4.0,
    Y_CACHE_TRACK + 0.3,
    "K/V Cache",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    fontstyle="italic",
    color=C_EDGE,
)

X_PREFILL_CENTER = 5.5 + 4.0 / 2
X_DECODE1_CENTER = 12.5 + 2.4 / 2
X_DECODE2_CENTER = 17.8 + 2.4 / 2

# --- 阶段 1: Prefill 并行写入缓存 ---
draw_cache_blocks(X_PREFILL_CENTER, Y_CACHE_TRACK, 3)
draw_arrow(
    X_PREFILL_CENTER - 0.6, Y_COMPUTE, X_PREFILL_CENTER - 0.6, Y_CACHE_TRACK + 0.6
)
draw_arrow(X_PREFILL_CENTER, Y_COMPUTE, X_PREFILL_CENTER, Y_CACHE_TRACK + 0.6)
draw_arrow(
    X_PREFILL_CENTER + 0.6, Y_COMPUTE, X_PREFILL_CENTER + 0.6, Y_CACHE_TRACK + 0.6
)

# --- 阶段 2: Decode 1 读写逻辑 ---
draw_cache_blocks(X_DECODE1_CENTER, Y_CACHE_TRACK, 4, is_new=True)
# 向上读取箭头：对齐左侧的历史块中心
draw_arrow(
    X_DECODE1_CENTER - 0.4, Y_CACHE_TRACK + 0.6, X_DECODE1_CENTER - 0.4, Y_COMPUTE
)
# 向下写入箭头：【精准计算】斜向指向第 4 个方块（红框，索引为3）的中心
new_block1_x = get_cache_block_center(X_DECODE1_CENTER, 4, target_idx=3)
draw_arrow(new_block1_x - 0.3, Y_COMPUTE, new_block1_x - 0.3, Y_CACHE_TRACK + 0.6)

# --- 阶段 3: Decode 2 读写逻辑 ---
draw_cache_blocks(X_DECODE2_CENTER, Y_CACHE_TRACK, 5, is_new=True)
# 向上读取箭头：对齐左侧的历史块中心
draw_arrow(
    X_DECODE2_CENTER - 0.4, Y_CACHE_TRACK + 0.6, X_DECODE2_CENTER - 0.4, Y_COMPUTE
)
# 向下写入箭头：【精准计算】斜向指向第 5 个方块（红框，索引为4）的中心
new_block2_x = get_cache_block_center(X_DECODE2_CENTER, 5, target_idx=4)
draw_arrow(new_block2_x - 0.4, Y_COMPUTE, new_block2_x - 0.4, Y_CACHE_TRACK + 0.6)

# ==================== 6. 渲染出图 ====================
plt.tight_layout()
plt.savefig("图2-1.pdf", bbox_inches="tight")
# plt.show()
