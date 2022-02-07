#include "test.cuh"

__global__ void test_kernel(void){}

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
    if(gpu_ptr) std::cout << "GPU address: " << gpu_ptr << std::endl;
    if(data_ptr) std::cout << "Data pointer: " << data_ptr << "offset: " << offset << std::endl;
}
