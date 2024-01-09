#include<cuda_runtime.h>

__global__ void matrixMul_native(const float *A, const float *B, const float *C, int M, int N, int K) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(ty >= M || tx >= N) {
        return;
    }

    for(int i = 0; i < N; i++) {
        C[tx][ty] += A[tx][i] * B[i][ty];
    }
}

int main() {
    float A_h[1024][1024], B_h[1024][1024];
}