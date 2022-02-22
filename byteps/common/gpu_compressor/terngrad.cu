#include "terngrad.cuh"
#include "math.h"

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void terngrad_compress(const void* gpu_ptr, size_t len, curandState *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float* ptr = reinterpret_cast<float*>(const_cast<void*>(gpu_ptr));
    float x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    float grad_max = fabsf(ptr[0]);
    float grad_abs;
    for(size_t i = 0; i < len; i++){
        grad_abs = fabsf(ptr[i]);
        if (grad_abs > grad_max) grad_max = grad_abs;
    }
    /* Generate pseudo-random uniforms */
    for(size_t i = 0; i < len; i++) {
        x = curand_uniform(&localState);
        if(x < fabsf(ptr[i])/grad_max) {
            if (ptr[i] > 0) ptr[i] = 1.0;
            else ptr[i] = -1.0;
        }
        else ptr[i] = 0.0;
    }
    /* Copy state back to global memory */
    state[id] = localState;
}

void compress(const void* gpu_ptr, size_t len){
    curandState *devStates;
    /* Allocate space for prng states on device */
    cudaMalloc((void**)&devStates, sizeof(curandState));
    /* Setup prng states */
    setup_kernel<<<1, 1>>>(devStates);
    terngrad_compress<<<1, 1>>>(gpu_ptr, len, devStates);
    /* Cleanup */
    cudaFree(devStates);
}

void decompress(const void* gpu_ptr, float scale, size_t len){
    // TODO: time the gradient with a scale
    // For now, just do nothing as I haven't figured out where to put scale
}
