import os, subprocess, time
import benchmark

os.makedirs("benchmark_results", exist_ok=True)

for k,v in benchmark.KERNEL_MAPS.items():
    print(f"\nRunning kernel {k}, {v[0]}...")
    cmd = f"python benchmark.py {k} | tee benchmark_results/{k}_{v[0]}_output.txt"
    subprocess.run(cmd, shell=True, check=True)
    time.sleep(2)

subprocess.run(["python", "plot_benchmark_results.py"], check=True)