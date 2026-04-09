文件说明：
1. flash_atten.py: 调用baseline+官方实现flash-attention+我实现的attention,在特定的长度下面的性能，结果写入到benchmark_results
2. benchmark_results: 下面存放各个实现的性能数据
2. src/runner.cu: 生成gridDim，blockDim，调用特定的kernel

环境：python3.12, torch 2.8(cuda 12.6), nvcc(cuda 12.6), 显卡驱动(cuda 13.0)
torch: pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
flash_attn: pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

对一个算子使用NVIDIA Nsight Compute分析时，要避免python、torch、nanobind的开销，单独写一个test_nuc.cu进行测试就可以了：
编译：nvcc test_ncu.cu -o test_ncu
测试：ncu -k rmsnorm_kernel ./test_ncu  # 生成一些摘要信息
     ncu -k rmsnorm_kernel -o rmsnorm_profile ./test_ncu # 生成.ncu-rep 报告文件，接着可以使用https://www.kapilsharma.dev/perfessor/进行可视化分析