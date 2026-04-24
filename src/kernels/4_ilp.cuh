#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 强制 8 字节对齐，生成极其高效的 LDG.64 指令
struct __align__(8) Pack64 {
  float2 data;
};

#include "common.cuh"

// ==========================================================
// 针对 1024 线程 (32个Warp) 量身定制的 Block 规约
// ==========================================================
__inline__ __device__ float blockReduceSum_1024(float val) {
  // 32 个 Warp，刚好需要 32 个 float 的 Shared Memory
  static __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  // 第一步：每个 Warp 内部规约
  val = warpReduceSum(val);

  // 第二步：每个 Warp 的 Leader 把结果写入 Shared Memory
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // 第三步：由第一个 Warp (Warp 0) 把这 32 个结果全部读出来并最后规约
  val = (threadIdx.x < 32) ? shared[threadIdx.x] : 0.0f;
  if (wid == 0) {
    val = warpReduceSum(val);
  }
  return val;
}

// ==========================================================
// 终极对齐 vLLM 的 RMSNorm
// HIDDEN_SIZE = 4096, 每个线程处理 4 个 half
// ==========================================================
__global__ void __launch_bounds__(1024) rms_norm_kernel_true_vllm(
    const half *__restrict__ input,  // [batch_size, 4096]
    const half *__restrict__ weight, // [4096]
    half *__restrict__ output,       // [batch_size, 4096]
    float epsilon) {
  // 每个线程负责 4 个 half (对应 1 个 Pack64 / float2)
  // 1024 threads * 4 = 4096 elements
  int tid = threadIdx.x;
  int row = blockIdx.x;

  int offset = row * 1024 + tid;

  const Pack64 *x_ptr = reinterpret_cast<const Pack64 *>(input);
  const Pack64 *w_ptr = reinterpret_cast<const Pack64 *>(weight);
  Pack64 *y_ptr = reinterpret_cast<Pack64 *>(output);

  // ==========================================================
  // 1. 发起 64-bit 并发加载 (打满 1024 线程的调度队列)
  // ==========================================================
  Pack64 x_vec = x_ptr[offset];
  const half2 *h2_x = reinterpret_cast<const half2 *>(&x_vec.data);

  // 2. 局部平方和计算 (FP32 累加保精度)
  float sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float2 f2_x = __half22float2(h2_x[i]);
    sum_sq += f2_x.x * f2_x.x + f2_x.y * f2_x.y;
  }

  // ==========================================================
  // 3. 32-Warp 规约
  // ==========================================================
  float total_sum_sq = blockReduceSum_1024(sum_sq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(total_sum_sq / 4096.0f + epsilon);
  }
  __syncthreads();

  // ==========================================================
  // 4. 读取 Weight、缩放并写回
  // ==========================================================
  half2 h2_inv_rms = __float2half2_rn(s_inv_rms);

  // 注意 Weight 是按 tid 读取，所有 Block 共享相同的 Cache 命中
  Pack64 w_vec = w_ptr[tid];
  const half2 *h2_w = reinterpret_cast<const half2 *>(&w_vec.data);

  Pack64 y_vec;
  half2 *h2_y = reinterpret_cast<half2 *>(&y_vec.data);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    // (X * inv_rms) * W
    half2 tmp = __hmul2(h2_x[i], h2_inv_rms);
    h2_y[i] = __hmul2(tmp, h2_w[i]);
  }

  y_ptr[offset] = y_vec;
}