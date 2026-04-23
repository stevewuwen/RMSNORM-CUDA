# Agent Instructions: rmsnorm_cuda

## Architecture & Boundaries
- This repository implements and benchmarks optimized CUDA kernels for RMSNorm (and related operations).
- C++ source code is in `src/`, with kernels defined in `src/kernels/`.
- The CUDA code is exposed to Python via a `nanobind` module named `rmsnorm_cuda`.
- Python scripts (e.g., `benchmark.py`, `benchmark_add_rmsnorm.py`) depend on this extension and expect it to be built. They append the `build/` directory to `sys.path` to import the module.

## Build Requirements
- The repository uses CMake to build the `nanobind` extension. You must rebuild it after modifying `.cu` or `.cuh` files.
- **Build command**: `cmake -B build -S . && cmake --build build -j$(nproc)`
- The extension artifact (`rmsnorm_cuda*.so`) must reside in the `build/` directory for the Python scripts to find it.

## Testing & Benchmarking
- **Run all benchmarks**: `python gen_benchmark_results.py`. This executes `benchmark.py` for all kernel variants, saves outputs to `benchmark_results/`, and automatically plots them.
- **Run a specific kernel benchmark**: `python benchmark.py <kernel_id>` (e.g., `python benchmark.py 3`).
- **Dependencies**: The benchmarks compare against `vllm`, `unsloth`, and `flash-attn`. See `README.md` for strict version requirements (e.g., `torch==2.8.0`, `vllm==0.11.0`). If compiling `unsloth`, note the `LD_LIBRARY_PATH` override required in `README.md` to avoid C++ environment issues.

## Profiling Quirks
- The codebase uses `torch.cuda.profiler.start()` and `stop()` inside the python scripts to isolate specific kernel measurements.
- When running NVIDIA Nsight Compute (`ncu`), you *must* use the `--profile-from-start off` flag to respect the Python profiler tags and avoid profiling Python/Torch overhead.
- **Example profiling command**: `ncu --profile-from-start off -o ncu_reports/my_report -f python benchmark.py <kernel_id>`
