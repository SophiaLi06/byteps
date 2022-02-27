#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define TOTAL_TIME_CUDA
//#define TIME_CUDA

void terngrad_compress(const void* gpu_ptr, size_t len);

void terngrad_decompress(const void* gpu_ptr, float scale, size_t len);
