#include "../src/kernels/1_native.cuh"
#include <cuda_runtime.h>
#include <iostream>

int main() {
  int num_rows = 4 * 4;
  int hidden_size = 4;
  int size = num_rows * hidden_size * sizeof(float);
  float *d_input, *d_weight, *d_output;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_weight, hidden_size * sizeof(float));
  cudaMalloc(&d_output, size);

  // Warmup
  rmsnorm_kernel<<<num_rows, 256>>>(d_input, d_weight, d_output, hidden_size,
                                    1e-6f);
  cudaDeviceSynchronize();

  // NCU profiling target
  rmsnorm_kernel<<<num_rows, 256>>>(d_input, d_weight, d_output, hidden_size,
                                    1e-6f);
  cudaDeviceSynchronize();

  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
  return 0;
}
