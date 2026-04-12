在A40设备上（理论内存带宽约696 GB/s），优化后的Kernel 4通过python benchmark.py 4测试可以达到 ~568.4 GB/s 的极高带宽，DRAM利用率在NCU报告中高达 *90.77%*。
这使得原本的基础Kernel版本（约499 GB/s）或者单纯依赖vec8的版本（约487 GB/s）的性能都得到了全面超越。
我所做的核心优化
基于NCU的报告（报告保存在了rmsnorm_native_v3.ncu-rep中），由于该Kernel属于典型的 DRAM_MEMORY_BOUND，我们对两处进行了彻底优化：
1. *Dynamic Shared Memory Caching (动态共享内存缓存)*：
   原来在计算平方和与最终乘加（FMA）时，对输入（input）各进行了一次全局DRAM读取。因此我们修改了runner.cu的rms_norm_kernel调用，为其分配了hidden_size * sizeof(half)的动态共享内存，在第一遍计算时不仅将其读入，且将其缓存进extern __shared__ half s_x[]中。这样在第二遍计算直接读取Shared Memory，把DRAM的读取量缩小了近一半。
   
2. *Vectorized Load/Store (向量化访存) + 循环展开 (Unroll)*：
   我们通过 reinterpret_cast<const float4*> 将 half 强转为 float4 类型，每次同时读取或写入 8 个 half，大大拓宽了单次总线吞吐量；同时配合 #pragma unroll 将转换与乘加完全展开为流水线形式，隐藏了指令延迟（ILP）。