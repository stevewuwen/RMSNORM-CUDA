import os
import re
import subprocess



def parse_and_calculate(output_text, kernel_id):
    """
    解析 NCU 输出，并格式化为指定格式。
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
    # 2. 提取 Shapes (修复部分)
    # 直接抓取 log 中的 "Dispatching Shape(B=..., Seq=..., Hidden=...)" 
    # --------------------------------------------------
    shape_matches = re.findall(
        r"Dispatching Shape\(B=\s*(\d+),\s*Seq=\s*(\d+),\s*Hidden=\s*(\d+)\)",
        output_text
    )

    parsed_shapes = []
    for b, s, h in shape_matches:
        # 按照目标格式重新格式化并对齐
        shape_str = f"Shape(B={int(b):<3}, Seq={int(s):>4}, Hidden={int(h)})"
        parsed_shapes.append(shape_str)

    # --------------------------------------------------
    # 3. 提取 metrics
    # --------------------------------------------------
    reads = re.findall(
        r"dram__bytes_read\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)",
        output_text
    )

    writes = re.findall(
        r"dram__bytes_write\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)",
        output_text
    )

    times = re.findall(
        r"gpu__time_duration\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)",
        output_text
    )

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
    # 输出整理
    # --------------------------------------------------
    lines = [
        header,
        "-" * 80
    ]

    count = min(
        len(parsed_shapes),
        len(reads),
        len(writes),
        len(times)
    )

    if count == 0:
        lines.append("No valid profiling data found.")
        return "\n".join(lines)

    for i in range(count):
        shape = parsed_shapes[i]

        r_unit, r_val = reads[i]
        w_unit, w_val = writes[i]
        t_unit, t_val = times[i]

        bytes_read = to_bytes(r_unit, r_val)
        bytes_write = to_bytes(w_unit, w_val)

        time_s = to_seconds(t_unit, t_val)
        time_us_val = to_us(t_unit, t_val)

        if time_s > 0:
            bw_gbps = ((bytes_read + bytes_write) / time_s / 1e9)
        else:
            bw_gbps = 0.0

        line = (
            f"{shape:<35} | "
            f"Time: {time_us_val:>7.2f} µs | "
            f"Bandwidth: {bw_gbps:>7.1f} GB/s"
        )

        lines.append(line)

    return "\n".join(lines)

def main():
    os.makedirs("benchmark_results", exist_ok=True)

    total_kernels = 2

    for kernel_id in range(total_kernels):
        print(
            f"[{kernel_id + 1}/{total_kernels}] "
            f"Profiling kernel {kernel_id}..."
        )

        cmd = [
            "ncu",
            "--profile-from-start", "off",
            "--metrics",
            "gpu__time_duration.sum,"
            "dram__bytes_read.sum,"
            "dram__bytes_write.sum",
            "python",
            "benchmark_add_rmsnorm.py",
            str(kernel_id)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            output = result.stdout + "\n" + result.stderr

            if result.returncode != 0:
                print(
                    f"Warning: kernel {kernel_id} "
                    f"returned code {result.returncode}"
                )

        except Exception as e:
            print(
                f"Failed kernel {kernel_id}: {str(e)}"
            )
            output = ""

        formatted_result = parse_and_calculate(
            output,
            kernel_id
        )

        out_file = (
            f"benchmark_results/"
            f"kernel_{kernel_id}_add_results.txt"
        )

        with open(
            out_file,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(formatted_result)
            f.write("\n")

        print(f"Saved → {out_file}\n")


if __name__ == "__main__":
    main()