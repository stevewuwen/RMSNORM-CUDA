#include "common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

// 修改为 Pack64：64 bits = float2 = 2 * half2 = 4 * half
union Pack64_add {
  float2 f2;
  half2 h2[2];
};

template <int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    add_rms_norm_kernel_optimized_pack64(
        const half *__restrict__ input,  // [bs, hidden_size]
        half *__restrict__ residual,     // [bs, hidden_size] (In-Out)
        const half *__restrict__ weight, // [hidden_size]
        half *__restrict__ output,       // [bs, hidden_size]
        int hidden_size, float epsilon) {
  // 假设 bs = gridDim.x，处理当前 row
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // 定位到当前行，并强转为 float2 指针以实现 64-bit 内存读写
  const float2 *x_row_f2 =
      reinterpret_cast<const float2 *>(input + row * hidden_size);
  float2 *r_row_f2 = reinterpret_cast<float2 *>(residual + row * hidden_size);
  const float2 *w_f2 = reinterpret_cast<const float2 *>(weight);
  float2 *y_row_f2 = reinterpret_cast<float2 *>(output + row * hidden_size);

  // 每次处理 4 个 half，计算总 Pack64 数量
  int num_f2 = hidden_size / 4; // 比如 4096 / 4 = 1024
  float partial_sum_sq = 0.0f;

  // 核心优化 1：使用寄存器数组替代 Shared Memory
  // 当 hs=4096, threads=256 时，num_f2=1024，每个线程刚好处理 4 个 float2。
  // 使用少量寄存器即可缓存 (Input+Residual) 的结果，彻底省去 Shared Memory
  float2 local_cache[8]; // 预留8个容量，最高支持 hs=8192@256threads

  // 第一阶段：Add(Input + Residual) -> 更新 Residual -> 计算平方和 ->
  // 存入寄存器
  int step = 0;
  for (int i = tid; i < num_f2; i += THREADS_PER_BLOCK) {
    Pack64_add x_pack, r_pack, r_new_pack;

    // 核心优化 2：利用 __ldg 强制走 L1 TEX Cache
    x_pack.f2 = __ldg(&x_row_f2[i]);
    r_pack.f2 = r_row_f2[i];

#pragma unroll
    for (int j = 0; j < 2; ++j) { // Pack64 只有 2 个 half2
      // FP16 向量化相加
      r_new_pack.h2[j] = __hadd2(x_pack.h2[j], r_pack.h2[j]);

      // 转成 FP32 计算平方和 (数值稳定性)
      float2 f2_val = __half22float2(r_new_pack.h2[j]);
      partial_sum_sq += f2_val.x * f2_val.x + f2_val.y * f2_val.y;
    }

    // 写回 Global Memory 的 Residual
    r_row_f2[i] = r_new_pack.f2;

    // 存入当前线程的本地寄存器 (Register Cache)
    local_cache[step] = r_new_pack.f2;
    step++;
  }

  // Block 级别的归约求和
  float total_sum_sq = blockReduceSum(partial_sum_sq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(total_sum_sq / hidden_size + epsilon);
  }
  __syncthreads();

  float inv_rms = s_inv_rms;

  // 第二阶段：从寄存器读取缓存结果，乘以 inv_rms 和 weight
  step = 0;
  for (int i = tid; i < num_f2; i += THREADS_PER_BLOCK) {
    Pack64_add r_new_pack, w_pack, out_pack;

    r_new_pack.f2 = local_cache[step];
    // weight 在所有 row 共享，使用 __ldg 命中 L1/L2 缓存
    w_pack.f2 = __ldg(&w_f2[i]);

#pragma unroll
    for (int j = 0; j < 2; ++j) { // Pack64 只有 2 个 half2
      float2 f2_v = __half22float2(r_new_pack.h2[j]);
      float2 f2_w = __half22float2(w_pack.h2[j]);

      f2_v.x = f2_v.x * inv_rms * f2_w.x;
      f2_v.y = f2_v.y * inv_rms * f2_w.y;

      out_pack.h2[j] = __float22half2_rn(f2_v);
    }
    y_row_f2[i] = out_pack.f2;
    step++;
  }
}