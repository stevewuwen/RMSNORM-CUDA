#include <cuda_runtime.h>
#pragma once
// ---------------------------------------------------------
// 1. Warp 级别归约求和
// ---------------------------------------------------------
inline __device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// ---------------------------------------------------------
// 2. Block 级别归约求和
// ---------------------------------------------------------
inline __device__ float blockReduceSum(float val) {
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