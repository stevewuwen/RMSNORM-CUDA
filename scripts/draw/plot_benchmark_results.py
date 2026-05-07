import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from pathlib import Path

# ================= 样式与字体设置优化 =================
# 使用更清爽明亮的白底网格主题
matplotlib.style.use("seaborn-v0_8-whitegrid")

# 设置中文字体（按优先级：微软雅黑(Win), 黑体(Win), 苹方(Mac), 默认无衬线）
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
# 修复中文字体下负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False

# 全局调大字体大小
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.facecolor": "white",
        "axes.titlesize": 24,  # 图表标题字体大小
        "axes.labelsize": 18,  # 坐标轴标签字体大小
        "xtick.labelsize": 14,  # X轴刻度数字大小
        "ytick.labelsize": 14,  # Y轴刻度数字大小
        "legend.fontsize": 14,  # 图例字体大小
        "legend.title_fontsize": 16,  # 图例标题字体大小
    }
)
# ======================================================


def parse_file(file):
    """
    The data we want to parse has this format:
    Running kernel 0: PyTorch_Pure_Python on cuda:0.
    --------------------------------------------------------------------------------
    Shape(B=4, Seq= 128, Hidden=4096) | Time:  118.34 µs | Bandwidth:   71.0 GB/s
    """
    with open(file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    data = {"batch_size": [], "bandwidth": [], "kernel_name": []}
    kernel_name = "Unknown"

    # 解析正则
    kernel_pattern = r"Running kernel \d+: (.*?) on"
    # 提取 B 和 Bandwidth (GB/s)
    data_pattern = r"Shape\(B=(\d+).*?\).*?Bandwidth:\s*([\d.]+)\s*GB/s"

    for line in lines:
        # 匹配 Kernel 名称
        if r := re.search(kernel_pattern, line):
            kernel_name = r.group(1).strip()
        # 匹配性能数据
        elif r := re.search(data_pattern, line):
            data["batch_size"].append(int(r.group(1)))
            data["bandwidth"].append(float(r.group(2)))
            data["kernel_name"].append(kernel_name)

    return data


def plot(df: pd.DataFrame):
    """
    The dataframe has 4 columns: kernel, batch_size, bandwidth, kernel_name
    """
    save_dir = Path.cwd()

    plt.figure(figsize=(18, 10))

    df["kernel_label"] = df.apply(
        lambda row: f"{row['kernel']}: {row['kernel_name']}", axis=1
    )

    # 1. 获取排序后的 kernel 列表，确保顺序一致
    unique_labels = sorted(
        df["kernel_label"].unique(), key=lambda x: int(x.split(":")[0])
    )

    # 2. 生成颜色，并创建一个严格的 {label: color} 映射字典
    colors = sn.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, colors))

    # 3. 绘制折线图与散点图 (合并在一起，并加粗线条和点)
    sn.lineplot(
        data=df,
        x="batch_size",
        y="bandwidth",
        hue="kernel_label",
        palette=color_map,
        hue_order=unique_labels,
        marker="o",  # 添加圆点标记
        markersize=10,  # 放大标记点
        linewidth=2,  # 加粗折线
    )

    # 设置 X 轴为对数坐标
    plt.xscale("log", base=2)
    from matplotlib.ticker import ScalarFormatter

    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    plt.xticks(df["batch_size"].unique(), df["batch_size"].unique())
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # 4. 图例外置及美化
    plt.legend(
        title="算子名称",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
        shadow=True,  # 给图例加个阴影更好看
    )

    # 5. 修改为中文文本，并调整边距
    plt.title("不同 RMSNorm 算子性能对比", pad=20, fontweight="bold")
    plt.xlabel("批次大小")
    plt.ylabel("显存带宽 (GB/s)")

    # 增加细微的网格线方便对齐数据
    plt.grid(True, which="major", linestyle="--", alpha=0.7)
    plt.grid(True, which="minor", linestyle=":", alpha=0.4)

    plt.tight_layout()

    plt.savefig(save_dir / "benchmark_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    results_dir = Path("benchmark_results")

    if not results_dir.exists():
        results_dir.mkdir()

    assert results_dir.is_dir(), "Benchmark results directory not found!"

    data = []
    for filename in results_dir.glob("*.txt"):
        stem_parts = filename.stem.split("_")
        if not stem_parts[0].isdigit():
            continue
        kernel_nr = int(stem_parts[0])
        results_dict = parse_file(filename)

        # 遍历解析出的 batch_size 和 bandwidth 列表
        for b_size, bw, k_name in zip(
            results_dict["batch_size"],
            results_dict["bandwidth"],
            results_dict["kernel_name"],
        ):
            data.append(
                {
                    "kernel": kernel_nr,
                    "batch_size": b_size,
                    "bandwidth": bw,
                    "kernel_name": k_name,
                }
            )

    if not data:
        print("No valid data found to plot.")
        exit(0)

    df = pd.DataFrame(data)
    plot(df)

    KERNEL_NAMES = (
        df.drop_duplicates("kernel").set_index("kernel")["kernel_name"].to_dict()
    )

    # 动态获取最大的 batch_size 作为 README 表格的数据对比
    max_batch_size = df["batch_size"].max()
    df_table = (
        df[df["batch_size"] == max_batch_size]
        .sort_values(by="bandwidth", ascending=True)[["kernel", "bandwidth"]]
        .copy()
    )

    df_table["kernel"] = df_table["kernel"].map(lambda k: f"{k}: {KERNEL_NAMES[k]}")

    base_idx = 0 if 0 in KERNEL_NAMES else min(KERNEL_NAMES.keys())
    baseline_kernel_name = f"{base_idx}: {KERNEL_NAMES[base_idx]}"

    baseline_bandwidth = df_table[df_table["kernel"] == baseline_kernel_name][
        "bandwidth"
    ].iloc[0]
    df_table["relperf"] = df_table["bandwidth"] / baseline_bandwidth
    df_table["relperf"] = df_table["relperf"].apply(lambda x: f"{x * 100:.1f}%")
    df_table.columns = [
        "Kernel",
        "Bandwidth (GB/s)",
        f"Performance relative to {KERNEL_NAMES[base_idx]}",
    ]

    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()

        if "<!-- benchmark_results -->" in readme:
            readme = re.sub(
                r"<!-- benchmark_results -->.*<!-- benchmark_results -->",
                "<!-- benchmark_results -->\n{}\n<!-- benchmark_results -->".format(
                    df_table.to_markdown(index=False)
                ),
                readme,
                flags=re.DOTALL,
            )
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme)
            print("Successfully updated README.md.")
        else:
            print("No <!-- benchmark_results --> tag found in README.md.")
            print(f"\n--- Markdown Table (Max Batch Size = {max_batch_size}) ---")
            print(df_table.to_markdown(index=False))
    else:
        print("README.md not found.")
        print(f"\n--- Markdown Table (Max Batch Size = {max_batch_size}) ---")
        print(df_table.to_markdown(index=False))
