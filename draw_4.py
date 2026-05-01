import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# 设置中文字体，防止中文显示乱码
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Songti SC",
    "Arial Unicode MS",
]  # 适配Win/Mac
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
ax.set_xlim(0, 160)
ax.set_ylim(0, 100)
ax.axis("off")  # 隐藏坐标轴

# 定义颜色
color_bus_bg = "#E0E0E0"  # 总线背景色（空载）
color_scalar = "#FF6B6B"  # 标量访存颜色 (红，警告)
color_pack64 = "#4D96FF"  # Pack64颜色 (蓝，均衡)
color_pack128 = "#6BCB77"  # Pack128颜色 (绿，极致)
color_text = "#333333"


# 绘制单个事务的函数
def draw_memory_transaction(
    y_pos,
    type_name,
    bit_width,
    total_reqs,
    util_rate,
    color,
    fp16_count,
    instruction,
    alignment,
    remark,
):
    # 1. 绘制128-bit总线/缓存行物理边界背景
    bus_width = 128
    bus_height = 10
    rect_bg = patches.Rectangle(
        (20, y_pos),
        bus_width,
        bus_height,
        linewidth=2,
        edgecolor="#999999",
        facecolor=color_bus_bg,
        linestyle="--",
    )
    ax.add_patch(rect_bg)

    # 绘制空载提示文字
    ax.text(
        20 + bus_width / 2,
        y_pos + bus_height / 2,
        "总线空载区域 (Wasted Bandwidth)",
        ha="center",
        va="center",
        color="#999999",
        fontsize=10,
        fontstyle="italic",
    )

    # 2. 绘制实际拉取的数据块 (分割为FP16)
    fp16_width = 16  # 16-bit per FP16
    for i in range(fp16_count):
        x_start = 20 + i * fp16_width
        rect_data = patches.Rectangle(
            (x_start, y_pos),
            fp16_width,
            bus_height,
            linewidth=1,
            edgecolor="white",
            facecolor=color,
        )
        ax.add_patch(rect_data)
        # FP16 标签
        ax.text(
            x_start + fp16_width / 2,
            y_pos + bus_height / 2,
            "FP16",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    # 3. 添加左侧标题
    ax.text(
        18,
        y_pos + bus_height / 2,
        type_name,
        ha="right",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=color_text,
    )

    # 4. 添加右侧/上方统计与说明信息
    info_x = 20 + bus_width + 5
    # 指令与对齐约束
    ax.text(
        20,
        y_pos + bus_height + 1.5,
        f"硬件指令: {instruction} | 约束: {alignment}",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#555555",
    )

    # 对比核心指标
    metrics_text = f"总线利用率: {util_rate}\n单次元素量: {fp16_count}个\n物理请求数: {total_reqs}\n特征: {remark}"
    ax.text(
        info_x,
        y_pos + bus_height / 2,
        metrics_text,
        ha="left",
        va="center",
        fontsize=11,
        color=color_text,
        bbox=dict(facecolor="#F8F9FA", edgecolor="#DDDDDD", boxstyle="round,pad=0.5"),
    )

    # 绘制一个拉取动作的示意框线
    if fp16_count < 8:
        ax.annotate(
            "",
            xy=(20 + fp16_count * fp16_width, y_pos - 2),
            xytext=(20, y_pos - 2),
            arrowprops=dict(arrowstyle="<->", color=color, lw=2),
        )
        ax.text(
            20 + (fp16_count * fp16_width) / 2,
            y_pos - 4,
            f"单次 {bit_width}-bit 事务",
            ha="center",
            va="top",
            color=color,
            fontsize=10,
        )
    else:
        ax.annotate(
            "",
            xy=(20 + bus_width, y_pos - 2),
            xytext=(20, y_pos - 2),
            arrowprops=dict(arrowstyle="<->", color=color, lw=2),
        )
        ax.text(
            20 + bus_width / 2,
            y_pos - 4,
            f"单次 {bit_width}-bit 事务 (填满Cache Line)",
            ha="center",
            va="top",
            color=color,
            fontsize=10,
        )


# ----------------- 开始绘制三层对比 -----------------

# 1. 标量访存 (最下层)
draw_memory_transaction(
    y_pos=15,
    type_name="标量访存\n(16-bit)",
    bit_width=16,
    total_reqs="极大 (基准)",
    util_rate="12.5%",
    color=color_scalar,
    fp16_count=1,
    instruction="LDG.16",
    alignment="2字节边界",
    remark="严重总线碎片化，指令发射压力大",
)

# 2. Pack64 向量化 (中间层)
draw_memory_transaction(
    y_pos=45,
    type_name="向量化 Pack64\n(64-bit / float2)",
    bit_width=64,
    total_reqs="降低 75%",
    util_rate="50.0%",
    color=color_pack64,
    fp16_count=4,
    instruction="LDG.64 (高效并发)",
    alignment="8字节边界 (较宽泛)",
    remark="资源与拉取平衡，SM调度最优解",
)

# 3. Pack128 向量化 (最上层)
draw_memory_transaction(
    y_pos=75,
    type_name="向量化 Pack128\n(128-bit / float4)",
    bit_width=128,
    total_reqs="降低 87.5%",
    util_rate="100% (满载)",
    color=color_pack128,
    fp16_count=8,
    instruction="LDG.128 (极致压缩)",
    alignment="16字节边界 (极严苛)",
    remark="需严防非对齐异常及Warp过少导致停滞",
)

# ----------------- 添加全局标注 -----------------

# 全局标题
plt.text(
    80,
    95,
    "图 4-1 16-bit标量访存与64/128-bit并发向量化访存对比图",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="black",
)

# 顶部物理内存虚线框标注
plt.text(
    84,
    88,
    "全局显存 (Global Memory) 128-bit 事务窗口 / Cache Line 物理边界",
    ha="center",
    va="center",
    fontsize=12,
    color="#777777",
    fontweight="bold",
)

# 左侧大括号/箭头指示优化方向
ax.annotate(
    "",
    xy=(5, 80),
    xytext=(5, 20),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=3),
)
plt.text(
    2,
    50,
    "向量化粒度提升\n总线载荷率增加",
    ha="right",
    va="center",
    rotation=90,
    fontsize=12,
    fontweight="bold",
    color="#333333",
)

# 保存及显示
plt.tight_layout()
plt.savefig("vectorization_memory_access_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
