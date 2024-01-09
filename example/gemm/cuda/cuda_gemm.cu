#include "cuda_gemm.h"

__global__ void gemm_native(float *a, float *b, float *c, int N, int M, int K){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.x;

    if(tx > M || ty > N)
        return;

    for(int i = 0; i < K; i++) {
        c[tx * M + ty] = a[tx * K + i] * b[i * N + ty];
    }
}

void gemm(float *a, float *b, float *c, int N, int M, int K){
    gemm_native(a, b, c,  N, M, K);
}