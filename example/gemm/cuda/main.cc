#include "cuda_gemm.h"

#include <iostream>

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

void printMatrix(float* matrix, int M, int N) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }

    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C_cublas = (float*)malloc(bytes_C);

    float *d_A;
    float *d_B;
    float *d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));

    for(int i = 0; i < M * K; i++) {
        h_A[i] = i;
    }

    for(int i = 0; i < K * N; i++) {
        h_B[i] = i + 1;
    }

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    gemm(d_A, d_B, d_C, N, M, K);

    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    printMatrix(h_A, M, K);
    printMatrix(h_B, K, N);
    printMatrix(h_C, M, N);

    return 0;
}