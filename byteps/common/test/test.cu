#include <cuda_runtime.h>
#include "test.cuh"
//#include <iostream>

__global__ void test_kernel(void){}

void test_wrapper(void){
    test_kernel <<<1, 1>>> ();
    std::cout << "Tested CUDA kernel" << std::endl;
}

void test_wrapper(void* gpu_ptr){
    test_kernel <<<1, 1>>> ();
    if(gpu_ptr) std::cout << "GPU address: " << gpu_ptr << std::endl;
}
