#include "cuda_gemm.h"

int upScale(int x, int y) {
    return (x + y - 1) / y;
}

__global__ void gemm_native(float *a, float *b, float *c, int N, int M, int K){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(tx >= M || ty >= N)
        return;

    int c_temp = 0;
    for(int i = 0; i < K; i++) {
        // c[tx * N + ty] += a[tx * K + i] * b[i * N + ty];
        c_temp += a[ty * K + i] * b[i * N + tx];  //结果一致
    }
    c[ty * N + tx] = c_temp;
}

__global__ void gemm_kernel_1(float *a, float *b, float *c, int N, int M, int K){
    int tx = blockIdx.x * blockDim.x + (threadIdx.x / 32);
    int ty = blockIdx.y * blockDim.x + (threadIdx.x % 32);

    if(tx >= M || ty >= N)
        return;

    int c_temp = 0;
    for(int i = 0; i < K; i++) {
        // c[tx * N + ty] += a[tx * K + i] * b[i * N + ty];
        c_temp += a[ty * K + i] * b[i * N + tx];  //结果一致
    }
    c[ty * N + tx] = c_temp;
}

// shared memory 
__global__ void gemm_kernel_2(float *a, float *b, float *c, int N, int M, int K){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // if(tx >= N || ty >= M)
    //     return;

    // shared memory
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int Ks = 32;
    float c_temp = 0;

    for(int i = 0; i < K / Ks; ++i) {
        As[threadIdx.y][threadIdx.x] = a[ty * K + i * Ks + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = b[i * Ks * N + tx + threadIdx.y * N];
        __syncthreads();

        for(int j = 0; j < Ks; ++j) {
            c_temp += As[threadIdx.y][j] * Bs[j][threadIdx.x]; 
        }
        __syncthreads();  
    }
    c[ty * N + tx] = c_temp;
}

inline void run_gemm_native(float *a, float *b, float *c, int N, int M, int K) {
    dim3 dimGrid(upScale(M, 32), upScale(N, 32));
    dim3 dimBlock(32, 32);

    gemm_native<<<dimGrid, dimBlock>>>(a, b, c, N, M, K);
}

void gemm(float *a, float *b, float *c, int N, int M, int K){
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_Y = 32;
    const int THREAD_SIZE_X = 32;

    dim3 dimBlock(THREAD_SIZE_X, THREAD_SIZE_Y);
    dim3 dimGrid(upScale(M, THREAD_SIZE_X), upScale(N, THREAD_SIZE_Y));
    // dim3 dimBlock(2, 2);
    // dim3 dimGrid(4, 4);

    int niter = 1000;
    float msecTotal = 0;
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    int kernel_num = 0;

    for(int i = 0; i < niter; i++) {
       gemm_kernel_2<<<dimGrid, dimBlock>>>(a, b, c, N, M, K); 
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    msecPerMatrixMul[0] = msecTotal / niter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("My gemm Performance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f Ops\n", 
            gigaFlops[0],
            msecPerMatrixMul[0],
            flopsPerMatrixMul);
    
}