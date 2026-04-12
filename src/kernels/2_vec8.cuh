// RMSNorm 这类操作在现代 GPU
// 上通常是访存密集型（Memory-Bound），而不是计算密集型。
// 为了达到极致的性能，我们需要从提升显存带宽利用率、指令级并行（ILP）以及减少冗余访存这三个核心角度进行优化。
// 以下是具体的优化思路及优化后的代码：
// 核心优化点分析：
// 向量化访存 (Vectorized Memory Access) —— 最关键的优化
// 原代码每次循环只读取一个 float（4 字节）。现代 GPU 显存总线非常宽，单次读取
// 16 字节 (float4) 能大幅提升全局内存的吞吐量。 大语言模型中 hidden_size
// 绝大多数情况下是 4 的倍数（如 4096, 8192），因此可以直接使用 float4
// 进行强制类型转换读取。 去除 static 关键字 原代码 blockReduceSum 中的 static
// __shared__ float shared[32];，static 是不必要的，直接写 __shared__ float
// shared[32]; 即可，避免引发语义上的误解。 指令级并行 (ILP) 在使用 float4
// 读取后，将 4 个元素的平方和计算展开，让编译器能更好地流水线化 FMA (Fused
// Multiply-Add) 指令，隐藏指令延迟。 利用 L2 Cache 避免寄存器溢出
// 理论上我们可以用寄存器把第一次读取的 x 存起来留给第二次计算用，但当
// hidden_size 很大时，寄存器会溢出到 Local
// Memory（极其缓慢）。因此我们保留两趟循环（Two-pass），利用 float4
// 极速读取，现代 GPU 的 L2 Cache 会极好地缓存第一趟读取的行数据。

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

  // 使用自定义的 Half8 结构体指针进行向量化读写 (单次访存 16 bytes)
  const Half8 *x_row_h8 =
      reinterpret_cast<const Half8 *>(input + row * hidden_size);
  const Half8 *w_row_h8 = reinterpret_cast<const Half8 *>(weight);
  Half8 *y_row_h8 = reinterpret_cast<Half8 *>(output + row * hidden_size);

  // 注意：此处 vec_size 是除以 8，因为我们一次处理 8 个 half
  int vec_size = hidden_size / 8;
  float partial_sum_sq = 0.0f;

  // 第一步：计算平方和
  for (int i = tid; i < vec_size; i += blockDim.x) {
    Half8 x = x_row_h8[i];

// 展开计算 4 个 half2
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // 必须转成 float 计算平方，否则 half 极易溢出
      float2 f2 = __half22float2(x.vals[j]);
      partial_sum_sq += f2.x * f2.x + f2.y * f2.y;
    }
  }

  // 第二步：Block 内规约
  float total_sum_sq = blockReduceSum(partial_sum_sq);

  // 第三步：计算 RMS 的倒数 (保持 float)
  __shared__ float s_inv_rms;
  if (tid == 0) {
    float mean_sq = total_sum_sq / hidden_size;
    s_inv_rms = rsqrtf(mean_sq + epsilon);
  }
  __syncthreads();

  // 第四步：应用 RMSNorm 并写回
  float inv_rms = s_inv_rms;
  for (int i = tid; i < vec_size; i += blockDim.x) {
    Half8 x = x_row_h8[i];
    Half8 w = w_row_h8[i];
    Half8 y;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // 转成 float 保证乘法精度
      float2 x_f2 = __half22float2(x.vals[j]);
      float2 w_f2 = __half22float2(w.vals[j]);
      float2 y_f2;

      y_f2.x = x_f2.x * inv_rms * w_f2.x;
      y_f2.y = x_f2.y * inv_rms * w_f2.y;

      // 计算完再转回 half2 存入结果中 (使用舍入到最近偶数模式)
      y.vals[j] = __float22half2_rn(y_f2);
    }

    // 一次性写回 8 个 half (16 Bytes)
    y_row_h8[i] = y;
  }
}