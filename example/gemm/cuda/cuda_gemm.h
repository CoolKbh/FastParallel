#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define checkCudaErrors(func) { \
    cudaError_t e = (func); \
    if(e != cudaSuccess) \
        printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
}

__global__ void gemm_native(float *a, float *b, float *c, int N, int M, int K);

void gemm(float *a, float *b, float *c, int N, int M, int K);