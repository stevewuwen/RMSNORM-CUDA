#include "common.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 使用 union 进行 float4 和 half2/half 的转换，避免 strict aliasing 导致的 UB
union Pack128 {
  float4 f4;
  half2 h2[4];
  half h[8];
};

__global__ void __launch_bounds__(1024, 1) add_rms_norm_kernel_shared_memory(
    const half *__restrict__ input,  // [num_rows, hidden_size]
    half *__restrict__ residual,     // [num_rows, hidden_size] (In-Out)
    const half *__restrict__ weight, // [hidden_size]
    half *__restrict__ output,       // [num_rows, hidden_size]
    int hidden_size, float epsilon) {

  int row = blockIdx.x;
  int tid = threadIdx.x;

  const half *x_row = input + row * hidden_size;
  half *r_row = residual + row * hidden_size;
  half *y_row = output + row * hidden_size;

  // 用于缓存 (input + residual) 的结果，避免第二次访问全局内存
  extern __shared__ half s_x[];

  float partial_sum_sq = 0.0f;
  int num_f4 = hidden_size / 8;

  const float4 *x_row_f4 = reinterpret_cast<const float4 *>(x_row);
  float4 *r_row_f4 = reinterpret_cast<float4 *>(r_row);
  float4 *s_x_f4 = reinterpret_cast<float4 *>(s_x);

  // 第一阶段：Add(Input + Residual) -> 更新全局 Residual -> 计算平方和 -> 存入
  // Shared Memory
  for (int i = tid; i < num_f4; i += blockDim.x) {
    Pack128 x_pack, r_pack, r_new_pack;
    x_pack.f4 = x_row_f4[i];
    r_pack.f4 = r_row_f4[i];

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // 1. FP16 向量化相加 (Input + Residual)
      r_new_pack.h2[j] = __hadd2(x_pack.h2[j], r_pack.h2[j]);

      // 2. 转成 FP32 计算平方和 (为了数值稳定性)
      float2 f2 = __half22float2(r_new_pack.h2[j]);
      partial_sum_sq += f2.x * f2.x + f2.y * f2.y;
    }

    // 3. 将相加后的最新 Residual 写回 Global Memory 和 Shared Memory
    r_row_f4[i] = r_new_pack.f4;
    s_x_f4[i] = r_new_pack.f4;
  }

  // 尾部处理 (非 8 的整数倍时)
  for (int i = num_f4 * 8 + tid; i < hidden_size; i += blockDim.x) {
    half x_val = x_row[i];
    half r_val = r_row[i];

    half r_new_val = __hadd(x_val, r_val);

    r_row[i] = r_new_val;
    s_x[i] = r_new_val;

    float val = __half2float(r_new_val);
    partial_sum_sq += val * val;
  }

  // Block 级别的归约求和
  float total_sum_sq = blockReduceSum(partial_sum_sq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    float mean_sq = total_sum_sq / hidden_size;
    s_inv_rms = rsqrtf(mean_sq + epsilon);
  }
  __syncthreads();

  float inv_rms = s_inv_rms;
  const float4 *w_f4 = reinterpret_cast<const float4 *>(weight);
  float4 *y_row_f4 = reinterpret_cast<float4 *>(y_row);

  // 第二阶段：读取 Shared Memory 缓存的残差结果，乘以 inv_rms 和 weight
  for (int i = tid; i < num_f4; i += blockDim.x) {
    Pack128 r_new_pack, w_pack, out_pack;
    r_new_pack.f4 = s_x_f4[i];
    w_pack.f4 = w_f4[i];

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2_v = __half22float2(r_new_pack.h2[j]);
      float2 f2_w = __half22float2(w_pack.h2[j]);

      f2_v.x = f2_v.x * inv_rms * f2_w.x;
      f2_v.y = f2_v.y * inv_rms * f2_w.y;

      out_pack.h2[j] = __float22half2_rn(f2_v);
    }
    y_row_f4[i] = out_pack.f4;
  }

  // 尾部处理
  for (int i = num_f4 * 8 + tid; i < hidden_size; i += blockDim.x) {
    y_row[i] =
        __float2half(__half2float(s_x[i]) * inv_rms * __half2float(weight[i]));
  }
}