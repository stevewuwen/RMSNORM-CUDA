#include "common.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
// 网格映射：一个 Block 处理一个 Row
__global__ void rms_norm_kernel(const half *__restrict__ input,
                                const half *__restrict__ weight,
                                half *__restrict__ output, int hidden_size,
                                float epsilon) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  const half *x_row = input + row * hidden_size;
  half *y_row = output + row * hidden_size;

  float partial_sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = __half2float(x_row[i]);
    partial_sum_sq += val * val;
  }

  float total_sum_sq = blockReduceSum(partial_sum_sq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    float mean_sq = total_sum_sq / hidden_size;
    s_inv_rms = rsqrtf(mean_sq + epsilon);
  }
  __syncthreads();
  float inv_rms = s_inv_rms;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    y_row[i] = __float2half(__half2float(x_row[i]) * inv_rms *
                            __half2float(weight[i]));
  }
}