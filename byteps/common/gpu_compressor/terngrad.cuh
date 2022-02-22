#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

void terngrad_compress(const void* gpu_ptr, size_t len);

void terngrad_decompress(const void* gpu_ptr, float scale, size_t len);
