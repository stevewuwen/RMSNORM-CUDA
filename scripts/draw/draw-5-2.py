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
        "lines.linewidth": 3.2,
    }
)
# ==================================================


def rename_kernel(kernel_name: str) -> str:
    """将原始 kernel 名称映射为论文图中的中文名称"""
    name_map = {
        "CUDA_NOT_FUSION_TLP": "未融合算子",
        "CUDA_FUSION_TLP": "融合算子",
        "PyTorch_Official_Compile": "PyTorch融合实现",
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
            if kernel_name == "Trit":
                kernel_name = "Triton_Custom"
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

    ax.set_xlabel("BatchSize", labelpad=10)

    ax.grid(True, which="major", linestyle="--", alpha=0.65)
    ax.grid(True, which="minor", linestyle=":", alpha=0.35)

    ax.tick_params(axis="both", which="major", pad=6)


def plot(df: pd.DataFrame):
    save_dir = Path.cwd()

    df = df.sort_values(["kernel_name", "batch_size"])
    df["kernel_label"] = df["kernel_name"]

    batch_sizes = sorted(df["batch_size"].unique())

    # 固定图例顺序，避免每次文件读取顺序不同导致图例顺序变化
    preferred_order = [
        "PyTorch算子",
        "Triton算子",
        "VLLM算子",
        "原生CUDA算子",
        "本文算子",
    ]

    existing_labels = list(df["kernel_label"].unique())
    kernel_labels = [x for x in preferred_order if x in existing_labels]
    kernel_labels += [x for x in existing_labels if x not in kernel_labels]

    colors = sn.color_palette("husl", len(kernel_labels))
    color_map = dict(zip(kernel_labels, colors))

    # 适当增高画布，给顶部图例留空间
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(28, 12),
        sharex=True,
    )

    ax_bw, ax_latency = axes

    # ================= 左图：带宽 =================
    for label in kernel_labels:
        sub_df = df[df["kernel_label"] == label].sort_values("batch_size")

        ax_bw.plot(
            sub_df["batch_size"],
            sub_df["bandwidth"],
            marker="o",
            markersize=11,
            linewidth=3.2,
            color=color_map[label],
            label=label,
        )

    ax_bw.set_title("显存带宽对比", pad=22, fontweight="bold")
    ax_bw.set_ylabel("显存带宽 (GB/s)", labelpad=12)
    format_x_axis(ax_bw, batch_sizes)

    # ================= 右图：延迟 =================
    for label in kernel_labels:
        sub_df = df[df["kernel_label"] == label].sort_values("batch_size")

        ax_latency.plot(
            sub_df["batch_size"],
            sub_df["latency_us"],
            marker="o",
            markersize=11,
            linewidth=3.2,
            color=color_map[label],
            label=label,
        )

    ax_latency.set_title("执行延迟对比", pad=22, fontweight="bold")
    ax_latency.set_ylabel("延迟 (µs)", labelpad=12)
    format_x_axis(ax_latency, batch_sizes)

    # ================= 顶部统一图例 =================
    handles, labels = ax_bw.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="算子名称",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=False,
        columnspacing=2.0,
        handlelength=2.8,
        borderpad=0.8,
        labelspacing=0.7,
    )

    # 手动控制布局：关键是 top 留出足够空间给图例
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.14,
        top=0.72,
        wspace=0.12,
    )

    fig.savefig(
        save_dir / "assets/图5-2.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
    )


if __name__ == "__main__":
    results_dir = Path("assets/addrmsnorm")

    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

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
