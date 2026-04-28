#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct __align__(8) Pack64 {
  float2 data;
};

#include "common.cuh"

__inline__ __device__ float blockReduceSum_1024(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceSum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < 32) ? shared[threadIdx.x] : 0.0f;
  if (wid == 0) {
    val = warpReduceSum(val);
  }
  return val;
}

__global__ void __launch_bounds__(1024)
    rms_norm_kernel_pack64(const half *__restrict__ input, // [batch_size, 4096]
                           const half *__restrict__ weight, // [4096]
                           half *__restrict__ output, // [batch_size, 4096]
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

  float total_sum_sq = blockReduceSum_1024(sum_sq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(total_sum_sq / 4096.0f + epsilon);
  }
  __syncthreads();

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