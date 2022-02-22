#include "terngrad.cuh"
#include "math.h"
#include <stdio.h>
#include <iostream>

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void find_grad_max(const void* gpu_ptr, size_t len, float* result){
    float* ptr = reinterpret_cast<float*>(const_cast<void*>(gpu_ptr));
    float grad_max;
    if (ptr[0] >= 0) grad_max = ptr[0];
    else grad_max = -ptr[0];
    float grad_abs;
    for(size_t i = 0; i < len; i++){
        if (ptr[i] >= 0) grad_abs = ptr[i];
        else grad_abs = -ptr[i];
        if (grad_abs > grad_max) grad_max = grad_abs;
    }
    *result = grad_max;
}

__global__ void terngrad_compress_kernel(const void* gpu_ptr, size_t len, curandState *state, float grad_max){
    //threadIdx.x contains the index of the current thread within its block, 
    //and blockDim.x contains the number of threads in the block
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float* ptr = reinterpret_cast<float*>(const_cast<void*>(gpu_ptr));
    float x;
    int index = threadIdx.x;
    int stride = blockDim.x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random uniforms */
    for(size_t i = index; i < len; i+=stride) {
        x = curand_uniform(&localState);
        if(x < fabsf(ptr[i])/grad_max) {
            if (ptr[i] > 0) ptr[i] = 1.0;
            else ptr[i] = -1.0;
        }
        else ptr[i] = 0.0;
        printf("Done index %d\n", i);
    }
    /* Copy state back to global memory */
    state[id] = localState;
}

void terngrad_compress(const void* gpu_ptr, size_t len){
    float* ptr = reinterpret_cast<float*>(const_cast<void*>(gpu_ptr));
    float grad_max;
    float* grad_max_answer;
    cudaMalloc(&grad_max_answer, sizeof(float));
    find_grad_max<<<1, 1>>>(gpu_ptr, len, grad_max_answer);
    cudaMemcpy(&grad_max, grad_max_answer, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(grad_max_answer);
    std::cout << "grad_max: " << grad_max << std::endl;
    
    const unsigned int threadsPerBlock = 64;
    // TODO: first try one block, then increase block number
    const unsigned int blockCount = 1;
    //const unsigned int blockCount = (len + threadsPerBlock - 1) / threadsPerBlock;
    const unsigned int totalThreads = threadsPerBlock * blockCount;
    curandState *devStates;
    // /* Allocate space for results on host */
    // hostResults = (unsigned int *)calloc(totalThreads, sizeof(int));
    /* Allocate space for prng states on device */
    cudaMalloc((void**)&devStates, totalThreads * sizeof(curandState));
    std::cout << "Done mallocing for devStates" << std::endl;
    /* Setup prng states */
    setup_kernel<<<blockCount, threadsPerBlock>>>(devStates);
    std::cout << "Done setup" << std::endl;
    terngrad_compress_kernel<<<blockCount, threadsPerBlock>>>(gpu_ptr, len, devStates, grad_max);
    /* Cleanup */
    cudaFree(devStates);
}

void terngrad_decompress(const void* gpu_ptr, float scale, size_t len){
    // TODO: time the gradient with a scale
    // For now, just do nothing as I haven't figured out where to put scale
}
