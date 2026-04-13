#include "common.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
// ---------------------------------------------------------
// RMSNorm Kernel
// ---------------------------------------------------------
// 网格映射：一个 Block 处理一个 Row (Token)
__global__ void __launch_bounds__(1024, 1) rms_norm_kernel_shared_memory(
    const half *__restrict__ input,  // [num_rows, hidden_size]
    const half *__restrict__ weight, // [hidden_size]
    half *__restrict__ output,       // [num_rows, hidden_size]
    int hidden_size, float epsilon) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // 指向当前行的起始位置
  const half *x_row = input + row * hidden_size;
  half *y_row = output + row * hidden_size;

  extern __shared__ half s_x[];

  // 第一步：计算当前线程负责的元素的平方和
  float partial_sum_sq = 0.0f;
  int num_f4 = hidden_size / 8;

  const float4 *x_row_f4 = reinterpret_cast<const float4 *>(x_row);
  float4 *s_x_f4 = reinterpret_cast<float4 *>(s_x);

  for (int i = tid; i < num_f4; i += blockDim.x) {
    float4 vals = x_row_f4[i];
    s_x_f4[i] = vals; // 缓存

    half2 *h2 = reinterpret_cast<half2 *>(&vals);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2 = __half22float2(h2[j]);
      partial_sum_sq += f2.x * f2.x + f2.y * f2.y;
    }
  }
  for (int i = num_f4 * 8 + tid; i < hidden_size; i += blockDim.x) {
    half x_val = x_row[i];
    s_x[i] = x_val;
    float val = __half2float(x_val);
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
  const float4 *w_f4 = reinterpret_cast<const float4 *>(weight);
  float4 *y_row_f4 = reinterpret_cast<float4 *>(y_row);

  for (int i = tid; i < num_f4; i += blockDim.x) {
    float4 vals = s_x_f4[i];
    float4 ws = w_f4[i];
    half2 *h2_vals = reinterpret_cast<half2 *>(&vals);
    const half2 *h2_ws = reinterpret_cast<const half2 *>(&ws);

    float4 out_vals;
    half2 *h2_out = reinterpret_cast<half2 *>(&out_vals);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2_v = __half22float2(h2_vals[j]);
      float2 f2_w = __half22float2(h2_ws[j]);
      f2_v.x = f2_v.x * inv_rms * f2_w.x;
      f2_v.y = f2_v.y * inv_rms * f2_w.y;
      h2_out[j] = __float22half2_rn(f2_v);
    }
    y_row_f4[i] = out_vals;
  }
  for (int i = num_f4 * 8 + tid; i < hidden_size; i += blockDim.x) {
    y_row[i] =
        __float2half(__half2float(s_x[i]) * inv_rms * __half2float(weight[i]));
  }
}