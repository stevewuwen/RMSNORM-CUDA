#include "kernels.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>

namespace nb = nanobind;

// CUDA 错误检查宏
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void launch_rmsnorm_native(const half *d_input, const half *d_weight,
                           half *d_output, int num_rows, int hidden_size,
                           float epsilon, cudaStream_t stream = 0) {
  // 每个 Token (Row) 分配一个 Block
  dim3 grid(num_rows);
  // Block 大小通常设为 256 或 512 即可，如果 hidden_size 很小可以设小点
  int block_size = 256;
  dim3 block(block_size);

  rms_norm_kernel<<<grid, block, 0, stream>>>(d_input, d_weight, d_output,
                                              hidden_size, epsilon);
  CHECK_CUDA(cudaGetLastError());
}

void launch_rmsnorm_vec8(const half *d_input, const half *d_weight,
                         half *d_output, int num_rows, int hidden_size,
                         float epsilon, cudaStream_t stream = 0) {
  // 每个 Token (Row) 分配一个 Block
  dim3 grid(num_rows);
  // Block 大小通常设为 256 或 512 即可，如果 hidden_size 很小可以设小点
  int block_size = 256;
  dim3 block(block_size);
  rms_norm_kernel_half_vec8<<<grid, block, 0, stream>>>(
      d_input, d_weight, d_output, hidden_size, epsilon);
  CHECK_CUDA(cudaGetLastError());
}

void launch_rmsnorm_shared_memory(const half *d_input, const half *d_weight,
                                  half *d_output, int num_rows, int hidden_size,
                                  float epsilon, cudaStream_t stream = 0) {
  // 每个 Token (Row) 分配一个 Block
  dim3 grid(num_rows);
  // 试试 1024
  int block_size = 1024;
  dim3 block(block_size);

  int shared_mem_size = hidden_size * sizeof(half);

  rms_norm_kernel_shared_memory<<<grid, block, shared_mem_size, stream>>>(
      d_input, d_weight, d_output, hidden_size, epsilon);
  CHECK_CUDA(cudaGetLastError());
}

void launch_rmsnorm_py(int kernel_num, nb::ndarray<nb::device::cuda> input,
                       nb::ndarray<nb::device::cuda> weight,
                       nb::ndarray<nb::device::cuda> output, float epsilon,
                       uintptr_t stream = 0) {
  int num_rows = 1;
  for (size_t i = 0; i < input.ndim() - 1; ++i) {
    num_rows *= input.shape(i);
  }
  int hidden_size = input.shape(input.ndim() - 1);
  switch (kernel_num) {
  case 6:
    launch_rmsnorm_native(static_cast<const half *>(input.data()),
                          static_cast<const half *>(weight.data()),
                          static_cast<half *>(output.data()), num_rows,
                          hidden_size, epsilon, (cudaStream_t)stream);
    break;
  case 7:
    launch_rmsnorm_vec8(static_cast<const half *>(input.data()),
                        static_cast<const half *>(weight.data()),
                        static_cast<half *>(output.data()), num_rows,
                        hidden_size, epsilon, (cudaStream_t)stream);
    break;
  case 8:
    launch_rmsnorm_shared_memory(static_cast<const half *>(input.data()),
                                 static_cast<const half *>(weight.data()),
                                 static_cast<half *>(output.data()), num_rows,
                                 hidden_size, epsilon, (cudaStream_t)stream);
  default:
    break;
  }
}

NB_MODULE(rmsnorm_cuda, m) {
  m.def("launch_rmsnorm", &launch_rmsnorm_py, "Launch RMSNorm CUDA kernel",
        nb::arg("kernel_num"), nb::arg("input"), nb::arg("weight"),
        nb::arg("output"), nb::arg("epsilon") = 1e-6, nb::arg("stream") = 0);
}
