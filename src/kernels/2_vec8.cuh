#include <cuda_fp16.h>

struct alignas(16) Half8 {
  half2 vals[4];
};

__global__ void rms_norm_kernel_half_vec8(
    const half *__restrict__ input,  // [num_rows, hidden_size]
    const half *__restrict__ weight, // [hidden_size]
    half *__restrict__ output,       // [num_rows, hidden_size]
    int hidden_size, float epsilon) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  const Half8 *x_row_h8 =
      reinterpret_cast<const Half8 *>(input + row * hidden_size);
  const Half8 *w_row_h8 = reinterpret_cast<const Half8 *>(weight);
  Half8 *y_row_h8 = reinterpret_cast<Half8 *>(output + row * hidden_size);

  int vec_size = hidden_size / 8;
  float partial_sum_sq = 0.0f;

  for (int i = tid; i < vec_size; i += blockDim.x) {
    Half8 x = x_row_h8[i];

// 展开计算 4 个 half2
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2 = __half22float2(x.vals[j]);
      partial_sum_sq += f2.x * f2.x + f2.y * f2.y;
    }
  }
  float total_sum_sq = blockReduceSum(partial_sum_sq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    float mean_sq = total_sum_sq / hidden_size;
    s_inv_rms = rsqrtf(mean_sq + epsilon);
  }
  __syncthreads();

  float inv_rms = s_inv_rms;
  for (int i = tid; i < vec_size; i += blockDim.x) {
    Half8 x = x_row_h8[i];
    Half8 w = w_row_h8[i];
    Half8 y;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 x_f2 = __half22float2(x.vals[j]);
      float2 w_f2 = __half22float2(w.vals[j]);
      float2 y_f2;

      y_f2.x = x_f2.x * inv_rms * w_f2.x;
      y_f2.y = x_f2.y * inv_rms * w_f2.y;
      y.vals[j] = __float22half2_rn(y_f2);
    }

    y_row_h8[i] = y;
  }
}