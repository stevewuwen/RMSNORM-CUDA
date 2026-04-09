#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// ---------------------------------------------------------
// 1. Warp 级别归约求和
// ---------------------------------------------------------
__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// ---------------------------------------------------------
// 2. Block 级别归约求和
// ---------------------------------------------------------
__inline__ __device__ float blockReduceSum(float val) {
  // 假设 blockDim.x 最大为 1024，最多 32 个 Warp
  static __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  // Warp 内规约
  val = warpReduceSum(val);

  // 每个 Warp 的第 0 个线程将结果写入 shared memory
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // 读取 shared memory 中的值到第一个 Warp，并进行最终规约
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

  if (wid == 0) {
    val = warpReduceSum(val);
  }

  return val;
}

// ---------------------------------------------------------
// 3. RMSNorm Kernel
// ---------------------------------------------------------
// 网格映射：一个 Block 处理一个 Row (Token)
__global__ void
rmsnorm_kernel(const float *__restrict__ input,  // [num_rows, hidden_size]
               const float *__restrict__ weight, // [hidden_size]
               float *__restrict__ output,       // [num_rows, hidden_size]
               int hidden_size, float epsilon) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // 指向当前行的起始位置
  const float *x_row = input + row * hidden_size;
  float *y_row = output + row * hidden_size;

  // 第一步：计算当前线程负责的元素的平方和
  float partial_sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = x_row[i];
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
    y_row[i] = x_row[i] * inv_rms * weight[i];
  }
}