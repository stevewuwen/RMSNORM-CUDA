import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# ================= 样式与字体设置 =================
matplotlib.style.use("seaborn-v0_8-whitegrid")

plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update(
    {
        "figure.dpi": 400,
        "savefig.facecolor": "white",
        "axes.titlesize": 38,
        "axes.labelsize": 32,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "legend.title_fontsize": 26,
    }
)
# ==================================================


def rename_kernel(kernel_name: str) -> str:
    """将原始 kernel 名称映射为论文图中的中文名称"""
    name_map = {
        "CUDA_TLP": "有三阶段无锁树形规约算法",
        "CUDA_TLP_Origin": "无三阶段无锁树形规约算法",
        "PyTorch_Pure_Python": "PyTorch 原生实现",
        "PyTorch_Official_Compile": "PyTorch Compile",
    }
    return name_map.get(kernel_name, kernel_name)


def parse_file(file: Path):
    """
    解析 benchmark 结果文件。

    示例格式：
    Running kernel 6: CUDA_TLP on cuda:0.
    Shape(B=1, Seq=   1, Hidden=4096) | Time:    4.29 µs | Bandwidth:    5.4 GB/s
    """
    with open(file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    data = {
        "batch_size": [],
        "latency_us": [],
        "bandwidth": [],
        "kernel_name": [],
    }

    kernel_name = "Unknown"

    kernel_pattern = r"Running kernel \d+:\s*(.*?)\s*on"
    data_pattern = (
        r"Shape\(B=(\d+).*?\)"
        r"\s*\|\s*Time:\s*([\d.]+)\s*[µu]s"
        r"\s*\|\s*Bandwidth:\s*([\d.]+)\s*GB/s"
    )

    for line in lines:
        kernel_match = re.search(kernel_pattern, line)
        if kernel_match:
            kernel_name = kernel_match.group(1).strip()
            continue

        data_match = re.search(data_pattern, line)
        if data_match:
            batch_size = int(data_match.group(1))
            latency_us = float(data_match.group(2))
            bandwidth = float(data_match.group(3))

            data["batch_size"].append(batch_size)
            data["latency_us"].append(latency_us)
            data["bandwidth"].append(bandwidth)
            data["kernel_name"].append(rename_kernel(kernel_name))

    return data


def format_x_axis(ax, batch_sizes):
    """统一设置 BatchSize 横坐标"""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels(batch_sizes, rotation=45, ha="right")
    ax.set_xlabel("BatchSize")

    ax.grid(True, which="major", linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)


def plot(df: pd.DataFrame):
    save_dir = Path.cwd()

    df = df.sort_values(["kernel_name", "batch_size"])
    df["kernel_label"] = df["kernel_name"]

    batch_sizes = sorted(df["batch_size"].unique())
    kernel_labels = list(df["kernel_label"].unique())

    colors = sn.color_palette("husl", len(kernel_labels))
    color_map = dict(zip(kernel_labels, colors))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(26, 10),
        sharex=True,
    )

    ax_bw, ax_latency = axes

    # ================= 左图：带宽 =================
    for label in kernel_labels:
        sub_df = df[df["kernel_label"] == label]

        ax_bw.plot(
            sub_df["batch_size"],
            sub_df["bandwidth"],
            marker="o",
            markersize=11,
            linewidth=3.2,
            color=color_map[label],
            label=label,
        )

    ax_bw.set_title("显存带宽对比", pad=18, fontweight="bold")
    ax_bw.set_ylabel("显存带宽 (GB/s)")
    format_x_axis(ax_bw, batch_sizes)

    # ================= 右图：延迟 =================
    for label in kernel_labels:
        sub_df = df[df["kernel_label"] == label]

        ax_latency.plot(
            sub_df["batch_size"],
            sub_df["latency_us"],
            marker="o",
            markersize=11,
            linewidth=3.2,
            color=color_map[label],
            label=label,
        )

    ax_latency.set_title("执行延迟对比", pad=18, fontweight="bold")
    ax_latency.set_ylabel("延迟 (µs)")
    format_x_axis(ax_latency, batch_sizes)

    # ================= 顶部统一图例 =================
    handles, labels = ax_bw.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="算子名称",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(len(labels), 3),
        frameon=True,
        shadow=True,
        columnspacing=1.8,
        handlelength=2.8,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig("assets/图5-3.pdf", bbox_inches="tight")


if __name__ == "__main__":
    results_dir = Path("assets/tree_result")

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

        for b_size, latency, bw, k_name in zip(
            results_dict["batch_size"],
            results_dict["latency_us"],
            results_dict["bandwidth"],
            results_dict["kernel_name"],
        ):
            data.append(
                {
                    "kernel": kernel_nr,
                    "batch_size": b_size,
                    "latency_us": latency,
                    "bandwidth": bw,
                    "kernel_name": k_name,
                }
            )

    if not data:
        print("No valid data found to plot.")
        exit(0)

    df = pd.DataFrame(data)
    plot(df)
