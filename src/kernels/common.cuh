#pragma once
#include <cuda_runtime.h>

// 1. Warp 级别归约求和 (使用更快的 XOR 蝶形规约)
inline __device__ float warpReduceSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// 2. Block 级别归约求和
inline __device__ float blockReduceSum(float val) {
  // 移除 static，避免潜在的跨 block 全局分配问题
  __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // 只有存在的 Warp 才去读
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;

  if (wid == 0) {
    val = warpReduceSum(val);
  }
  return val;
}