#include <cuda_runtime.h>

__global__ void test_kernel(void){}

extern "C" void test_wrapper(void){
    test_kernel <<<1, 1>>> ();
    std::cout << "Tested CUDA kernel" << std::endl;
}
