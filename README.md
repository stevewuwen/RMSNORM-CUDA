### 文件说明
1. flash_atten.py: 调用baseline+官方实现flash-attention+我实现的attention,在特定的长度下面的性能，结果写入到benchmark_results
2. benchmark_results: 下面存放各个实现的性能数据
2. src/runner.cu: 生成gridDim，blockDim，调用特定的kernel

### 环境说明
环境：python3.12, torch 2.8(cuda 12.6), nvcc(cuda 12.6), 显卡驱动(cuda 13.0)
torch: pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
flash_attn: pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
vllm: pip install vllm==0.11.0 (只支持torch2.8)
在编译unsloth时需要修改c++环境：
export LD_LIBRARY_PATH=/home/wenjun/miniconda3/envs/ATTN_CUDA/lib:$LD_LIBRARY_PATH

### ncu分析
对一个算子使用NVIDIA Nsight Compute分析时，要避免python、torch、nanobind的开销，因此要在预热后显示开启分析和结束分析，让其只分析一个kernel：
```python
# 开启分析标记
torch.cuda.synchronize()
torch.cuda.profiler.start()  # 通知 ncu 开始记录

run_rms_norm_kernel(x, weight)

torch.cuda.synchronize()
torch.cuda.profiler.stop()   # 通知 ncu 停止记录
```
接着使用下面的命令进行记录：
```bash
ncu --profile-from-start off -o my_profile_report     -f     python benchmark.py 3
```

--profile-from-start off: 告诉 ncu 启动程序后先“挂起”采集功能，直到遇到代码中的 cudaProfilerStart() 才开始写数据。
--kernel-name-base regex:rms_norm: 使用正则匹配你的 kernel 名字，避免分析到 PyTorch 自带的初始化 kernel。
--launch-count 1: 只记录匹配到的第一个 kernel 实例。
-f: 强制覆盖已有的报告文件。

可以在指定网站查看对应的ncu-rep文件的分析：