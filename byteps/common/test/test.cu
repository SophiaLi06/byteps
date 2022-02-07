#include "test.cuh"
#include <stdio.h>

// assume the data type is float 32 for now. In the future, better do dtype check
__global__ void test_kernel(void){}

__global__ void test_mul(const void* p, size_t len){
    float* ptr = reinterpret_cast<float*>(const_cast<void*>(p));
    for(size_t i = 0; i < len; i++) {
        //printf("Before %f\n", ptr[i]);
        ptr[i] = ptr[i] * 2;
        //printf("After %f\n", ptr[i]);
    }
}

void test_wrapper(void){
    test_kernel <<<1, 1>>> ();
    std::cout << "Tested CUDA kernel" << std::endl;
}

void test_wrapper(void* gpu_ptr){
    test_kernel <<<1, 1>>> ();
    if(gpu_ptr) std::cout << "GPU address: " << gpu_ptr << std::endl;
}

void test_wrapper(void* gpu_ptr, const void* data_ptr, int offset){
    test_kernel <<<1, 1>>> ();
    //if(gpu_ptr) std::cout << "GPU address: " << gpu_ptr << std::endl;
    //if(data_ptr) std::cout << "Data pointer: " << data_ptr << "offset: " << offset << std::endl;
}

void test_mul_wrapper(const void* p, size_t len){
    test_mul <<<1, 1>>> (p, len);
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    //std::cout << "Tested CUDA multiply" << std::endl;
}