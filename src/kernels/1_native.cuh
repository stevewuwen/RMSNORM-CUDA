#include "common.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
// ---------------------------------------------------------
// 3. RMSNorm Kernel
// ---------------------------------------------------------
// 网格映射：一个 Block 处理一个 Row (Token)
__global__ void
rms_norm_kernel(const half *__restrict__ input,  // [num_rows, hidden_size]
                const half *__restrict__ weight, // [hidden_size]
                half *__restrict__ output,       // [num_rows, hidden_size]
                int hidden_size, float epsilon) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // 指向当前行的起始位置
  const half *x_row = input + row * hidden_size;
  half *y_row = output + row * hidden_size;

  // 第一步：计算当前线程负责的元素的平方和
  float partial_sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = __half2float(x_row[i]);
    partial_sum_sq += val * val;
  }

  // 第二步：Block 内规约，得到总平方和
  float total_sum_sq = blockReduceSum(partial_sum_sq);

  // 第三步：计算 RMS 的倒数 (Inverse RMS)
  __shared__ float s_inv_rms;
  if (tid == 0) {
    float mean_sq = total_sum_sq / hidden_size;
    // rsqrtf 是 CUDA 原生的快速平方根倒数指令
    s_inv_rms = rsqrtf(mean_sq + epsilon);
  }
  __syncthreads(); // 确保所有线程都能读取到 s_inv_rms

  // 第四步：应用 RMSNorm 并乘上可学习参数 (weight)
  float inv_rms = s_inv_rms;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    y_row[i] = __float2half(__half2float(x_row[i]) * inv_rms *
                            __half2float(weight[i]));
  }
}