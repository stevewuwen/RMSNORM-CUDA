#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.cuh"

// 【修改点】：替换为普通/经典的共享内存树状规约算法
__inline__ __device__ float blockReduceSum_ordinary(float val) {
  // 普通规约需要为每个线程分配一个共享内存空间（针对1024个线程）
  __shared__ float shared[1024];
  int tid = threadIdx.x;

  // 第一步：将所有线程的局部结果写入共享内存
  shared[tid] = val;
  __syncthreads(); // 等待所有线程写入完成

// 第二步：标准的二叉树折叠规约
// stride 从 512 (1024/2) 开始，每次减半直到 1
#pragma unroll
  for (int stride = 512; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    // 【性能痛点】：普通规约在每一次折叠后都必须强制同步整个Block
    __syncthreads();
  }

  // 规约最终结果保存在 shared[0] 中
  return shared[0];
}

__global__ void __launch_bounds__(1024) rms_norm_kernel_pack64_origin(
    const half *__restrict__ input,  // [batch_size, 4096]
    const half *__restrict__ weight, // [4096]
    half *__restrict__ output,       // [batch_size, 4096]
    float epsilon) {
  int tid = threadIdx.x;
  int row = blockIdx.x;

  int offset = row * 1024 + tid;

  const Pack64 *x_ptr = reinterpret_cast<const Pack64 *>(input);
  const Pack64 *w_ptr = reinterpret_cast<const Pack64 *>(weight);
  Pack64 *y_ptr = reinterpret_cast<Pack64 *>(output);

  Pack64 x_vec = x_ptr[offset];
  const half2 *h2_x = reinterpret_cast<const half2 *>(&x_vec.data);

  float sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float2 f2_x = __half22float2(h2_x[i]);
    sum_sq += f2_x.x * f2_x.x + f2_x.y * f2_x.y;
  }

  // 【修改点】：调用普通的规约函数
  float total_sum_sq = blockReduceSum_ordinary(sum_sq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(total_sum_sq / 4096.0f + epsilon);
  }
  __syncthreads(); // 这里的同步用于确保 s_inv_rms 被计算完毕并广播给所有线程

  half2 h2_inv_rms = __float2half2_rn(s_inv_rms);

  Pack64 w_vec = w_ptr[tid];
  const half2 *h2_w = reinterpret_cast<const half2 *>(&w_vec.data);

  Pack64 y_vec;
  half2 *h2_y = reinterpret_cast<half2 *>(&y_vec.data);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    half2 tmp = __hmul2(h2_x[i], h2_inv_rms);
    h2_y[i] = __hmul2(tmp, h2_w[i]);
  }

  y_ptr[offset] = y_vec;
}