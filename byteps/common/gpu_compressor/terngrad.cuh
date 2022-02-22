#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

void compress(const void* gpu_ptr, size_t len, curandState *state);

void decompress(const void* gpu_ptr. float scale. size_t len);