import os
import re
import subprocess

def parse_and_calculate(output_text, kernel_id):
    # 1. 提取头部内核信息 (例如: Running kernel 6: CUDA_Native on cuda:0.)
    header_match = re.search(r'(Running kernel \d+: .* on cuda:\d+\.)', output_text)
    header = header_match.group(1) if header_match else f"Running kernel {kernel_id}: Unknown on cuda:0."
    
    # 2. 提取所有的 Shape 参数，保持原始空格格式
    shapes = re.findall(r'Dispatching (Shape\(.*?\)) for NCU', output_text)
    
    # 3. 提取所有 NCU Profile 指标 (单位和数值)
    reads = re.findall(r'dram__bytes_read\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)', output_text)
    writes = re.findall(r'dram__bytes_write\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)', output_text)
    times = re.findall(r'gpu__time_duration\.sum\s+([a-zA-Z]+)\s+([\d\.,]+)', output_text)
    
    # 单位转换辅助函数
    def to_bytes(unit, val):
        val = float(val.replace(',', ''))
        unit = unit.lower()
        if 'k' in unit: return val * 1024
        if 'm' in unit: return val * 1024**2
        if 'g' in unit: return val * 1024**3
        return val # byte
        
    def to_seconds(unit, val):
        val = float(val.replace(',', ''))
        unit = unit.lower()
        if unit == 'us': return val * 1e-6
        if unit == 'ms': return val * 1e-3
        if unit == 's': return val
        return val
        
    def to_us(unit, val):
        val = float(val.replace(',', ''))
        unit = unit.lower()
        if unit == 'us': return val
        if unit == 'ms': return val * 1000
        if unit == 's': return val * 1e6
        return val

    # 准备输出行
    lines = [header, "-" * 80]
    
    # 由于可能存在 NCU 初始化输出，我们取匹配到的最小数量进行 zip
    count = min(len(shapes), len(reads), len(writes), len(times))
    
    for i in range(count):
        shape = shapes[i]
        
        r_unit, r_val = reads[i]
        w_unit, w_val = writes[i]
        t_unit, t_val = times[i]
        
        bytes_read = to_bytes(r_unit, r_val)
        bytes_write = to_bytes(w_unit, w_val)
        time_s = to_seconds(t_unit, t_val)
        time_us_val = to_us(t_unit, t_val)
        
        # 计算有效带宽 GB/s (1 GB = 10^9 Bytes)
        if time_s > 0:
            bw_gbps = (bytes_read + bytes_write) / time_s / 1e9
        else:
            bw_gbps = 0.0
            
        # 格式化输出 (保持和需求一样对齐)
        line = f"{shape} | Time: {time_us_val:>7.2f} µs | Bandwidth: {bw_gbps:>6.1f} GB/s"
        lines.append(line)
        
    return "\n".join(lines)

def main():
    # 创建结果保存目录
    os.makedirs("benchmark_results", exist_ok=True)
    
    kernel_ids = [6]
    for kernel_id in kernel_ids:
        print(f"[{kernel_id}/6] Profiling kernel {kernel_id}...")
        
        # 构建执行命令
        cmd = [
            "ncu", "--profile-from-start", "off",
            "--metrics", "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum",
            "python", "benchmark_more_batch_size.py", str(kernel_id)
        ]
        
        # 运行子进程并捕获标准输出和标准错误
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout + "\n" + result.stderr
        except subprocess.CalledProcessError as e:
            print(f"Error running kernel {kernel_id}!")
            output = e.stdout + "\n" + e.stderr
            
        # 解析输出并计算数据
        formatted_result = parse_and_calculate(output, kernel_id)
        
        # 将结果写入对应文件 (使用 utf-8 确保 µs 等特殊符号正常保存)
        out_file = f"benchmark_results/{kernel_id}_more_batch_size_results_kernel.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(formatted_result)
            f.write("\n")
            
        print(f"Saved results to {out_file}\n")

if __name__ == '__main__':
    main()