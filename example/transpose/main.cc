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
        h_A[i] = i / 13;
    }

    for(int i = 0; i < K * N; i++) {
        h_B[i] = i % 13;
    }

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    gemm(d_A, d_B, d_C, N, M, K);

    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // printMatrix(h_A, M, K);
    // printMatrix(h_B, K, N);
    // printMatrix(h_C, M, N);

    // cublas

    // double msecPerMatrixMul[2] = {0, 0};
    // double gigaFlops[2] = {0, 0};
    // double flopsPerMatrixMul = 2.0 * M * N * K;

    // int nIter = 1000;
    // float msecTotal = 0;
    // cudaEvent_t start, stop;
    // checkCudaErrors(cudaEventCreate(&start));
    // checkCudaErrors(cudaEventCreate(&stop));

    // cublasHandle_t blas_handle;  
    // cublasCreate(&blas_handle);
    // float alpha = 1.0;
    // float beta = 0;
    // checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaEventRecord(start));
    // for (int run = 0 ; run < nIter; run ++ ) {
    //     cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
    //         M, N, K, &alpha, 
    //         d_A, K, d_B, N, &beta, d_C, N
    //     );
    // }
    // checkCudaErrors(cudaEventRecord(stop));
    // checkCudaErrors(cudaEventSynchronize(stop));
    // checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // checkCudaErrors(cudaMemcpy( h_C_cublas, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // msecPerMatrixMul[1] = msecTotal / nIter;
    // gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    // printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
    //     gigaFlops[1],
    //     msecPerMatrixMul[1],
    //     flopsPerMatrixMul);

    // cublasDestroy(blas_handle); 
    
    // double eps = 1.e-6;  // machine zero
    // bool correct = true;
    // for (int i = 0; i < M * N; i++) {
    //     int row = i / N;
    //     int col = i % N;
    //     double abs_err = fabs(h_C[i] - h_C_cublas[col * M + row]);
    //     double dot_length = M;
    //     double abs_val = fabs(h_C[i]);
    //     double rel_err = abs_err / abs_val / dot_length;
    //     if (rel_err > eps) {
    //         printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
    //                 i, h_C[i], h_C_cublas[col * M + row], eps);
    //         correct = false;
    //         break;
    //     }
    // }

    // printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    // printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

    return 0;
}