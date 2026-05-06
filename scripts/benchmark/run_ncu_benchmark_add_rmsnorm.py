import os
import re
import subprocess

def parse_and_calculate(output_text, kernel_id):
    """
    解析 NCU 输出，并格式化为指定格式（支持多个 Kernel 组合为一次操作）。
    """

    # --------------------------------------------------
    # 1. 提取头部信息
    # --------------------------------------------------
    header_match = re.search(
        r"(Running kernel \d+: .* on cuda:\d+\.)",
        output_text
    )

    if header_match:
        header = header_match.group(1)
    else:
        header = f"Running kernel {kernel_id}: Unknown on cuda:0."

    # --------------------------------------------------
    # 2. 提取 Shapes
    # --------------------------------------------------
    shape_matches = re.findall(
        r"Dispatching Shape\(B=\s*(\d+),\s*Seq=\s*(\d+),\s*Hidden=\s*(\d+)\)",
        output_text
    )

    parsed_shapes = []
    for b, s, h in shape_matches:
        shape_str = f"Shape(B={int(b):<3}, Seq={int(s):>4}, Hidden={int(h)})"
        parsed_shapes.append(shape_str)

    # --------------------------------------------------
    # 3. 提取 metrics
    # --------------------------------------------------
    reads = re.findall(r"dram__bytes_read\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)", output_text)
    writes = re.findall(r"dram__bytes_write\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)", output_text)
    times = re.findall(r"gpu__time_duration\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)", output_text)

    # --------------------------------------------------
    # 单位转换
    # --------------------------------------------------
    def to_bytes(unit, val):
        val = float(val.replace(",", ""))
        unit = unit.lower()
        if unit == "byte": return val
        elif unit == "kbyte": return val * 1024
        elif unit == "mbyte": return val * 1024 ** 2
        elif unit == "gbyte": return val * 1024 ** 3
        else: return val

    def to_seconds(unit, val):
        val = float(val.replace(",", ""))
        unit = unit.lower()
        if unit == "us": return val * 1e-6
        elif unit == "ms": return val * 1e-3
        elif unit == "s": return val
        else: return val

    def to_us(unit, val):
        val = float(val.replace(",", ""))
        unit = unit.lower()
        if unit == "us": return val
        elif unit == "ms": return val * 1000
        elif unit == "s": return val * 1e6
        else: return val

    # --------------------------------------------------
    # 输出整理 (修复部分：支持多 Kernel 聚合)
    # --------------------------------------------------
    lines = [
        header,
        "-" * 80
    ]

    num_shapes = len(parsed_shapes)
    num_metrics = min(len(reads), len(writes), len(times))

    if num_shapes == 0 or num_metrics == 0:
        lines.append("No valid profiling data found.")
        return "\n".join(lines)

    # 动态计算每次 Shape 包含几个 Kernel (例如 16个metrics // 8个shapes = 2)
    kernels_per_shape = num_metrics // num_shapes

    if num_metrics % num_shapes != 0:
        lines.append(f"Warning: Unmatched shapes and kernels. Shapes: {num_shapes}, Kernels: {num_metrics}")

    for i in range(num_shapes):
        shape = parsed_shapes[i]
        
        total_bytes_read = 0.0
        total_bytes_write = 0.0
        total_time_s = 0.0
        total_time_us = 0.0

        # 将属于同一个 Shape 的多个 Kernel 的数据进行累加
        for k in range(kernels_per_shape):
            idx = i * kernels_per_shape + k
            
            r_unit, r_val = reads[idx]
            w_unit, w_val = writes[idx]
            t_unit, t_val = times[idx]

            total_bytes_read += to_bytes(r_unit, r_val)
            total_bytes_write += to_bytes(w_unit, w_val)
            total_time_s += to_seconds(t_unit, t_val)
            total_time_us += to_us(t_unit, t_val)

        # 基于累加的总时间与总数据量计算综合带宽
        if total_time_s > 0:
            bw_gbps = ((total_bytes_read + total_bytes_write) / total_time_s / 1e9)
        else:
            bw_gbps = 0.0

        line = (
            f"{shape:<35} | "
            f"Time: {total_time_us:>7.2f} µs | "
            f"Bandwidth: {bw_gbps:>7.1f} GB/s"
        )

        lines.append(line)

    return "\n".join(lines)


def main():
    os.makedirs("benchmark_results", exist_ok=True)

    total_kernels = 5

    for kernel_id in range(total_kernels):
        print(f"[{kernel_id + 1}/{total_kernels}] Profiling kernel {kernel_id}...")

        cmd = [
            "ncu",
            "--profile-from-start", "off",
            "--metrics",
            "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum",
            "python",
            "scripts/benchmark/benchmark_add_rmsnorm.py",
            str(kernel_id)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout + "\n" + result.stderr

            if result.returncode != 0:
                print(f"Warning: kernel {kernel_id} returned code {result.returncode}")

        except Exception as e:
            print(f"Failed kernel {kernel_id}: {str(e)}")
            output = ""

        formatted_result = parse_and_calculate(output, kernel_id)
        out_file = f"benchmark_results/{kernel_id}_kernel_add_results.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(formatted_result)
            f.write("\n")

        print(f"Saved → {out_file}\n")


if __name__ == "__main__":
    main()