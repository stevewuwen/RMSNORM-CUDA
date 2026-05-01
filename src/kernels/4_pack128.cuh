#include "common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

template <int THREADS_PER_BLOCK = 256>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    rms_norm_kernel_pack128(const half *__restrict__ input, // [bs, hidden_size]
                            const half *__restrict__ weight, // [hidden_size]
                            half *__restrict__ output, // [bs, hidden_size]
                            int hidden_size, float epsilon) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  const float4 *x_row_f4 =
      reinterpret_cast<const float4 *>(input + row * hidden_size);
  const float4 *w_f4 = reinterpret_cast<const float4 *>(weight);
  float4 *y_row_f4 = reinterpret_cast<float4 *>(output + row * hidden_size);

  int num_packs = hidden_size / 8; // 4096 / 8 = 512 次 128-bit 访存
  float partial_sum_sq = 0.0f;

  float4 local_cache[4];
  int step = 0;

  for (int i = tid; i < num_packs; i += THREADS_PER_BLOCK) {
    Pack128 x_pack;
    x_pack.f4 = __ldg(&x_row_f4[i]);

    // 存入当前线程的寄存器
    local_cache[step++] = x_pack.f4;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2 = __half22float2(x_pack.h2[j]);
      partial_sum_sq += f2.x * f2.x + f2.y * f2.y;
    }
  }

  float total_sum_sq = blockReduceSum(partial_sum_sq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(total_sum_sq / hidden_size + epsilon);
  }
  __syncthreads();

  float inv_rms = s_inv_rms;

  step = 0;
  for (int i = tid; i < num_packs; i += THREADS_PER_BLOCK) {
    Pack128 x_pack, w_pack, out_pack;

    // 直接从本地寄存器拿 Input (0 延迟)
    x_pack.f4 = local_cache[step++];

    w_pack.f4 = __ldg(&w_f4[i]);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2_x = __half22float2(x_pack.h2[j]);
      float2 f2_w = __half22float2(w_pack.h2[j]);

      // (X * inv_rms) * W
      f2_x.x = (f2_x.x * inv_rms) * f2_w.x;
      f2_x.y = (f2_x.y * inv_rms) * f2_w.y;

      out_pack.h2[j] = __float22half2_rn(f2_x);
    }

    y_row_f4[i] = out_pack.f4;
  }
}