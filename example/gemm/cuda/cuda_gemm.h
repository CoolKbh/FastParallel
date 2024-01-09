#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void gemm_native(float *a, float *b, float *c, int N, int M, int K);

void gemm(float *a, float *b, float *c, int N, int M, int K);