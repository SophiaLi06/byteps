#include <cuda_runtime.h>
#include <iostream>

void test_wrapper(void);

void test_wrapper(void* gpu_ptr);

void test_wrapper(void* gpu_ptr, const void* data_ptr, int offset);