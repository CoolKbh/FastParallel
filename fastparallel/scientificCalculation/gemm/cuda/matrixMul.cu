#include <cuda_runtime.h>

template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_Y,
    const int THREAD_SIZE_X,
    const bool ENABLE_DOUBLE_BUFFER>
__global__ void sgemm()
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadId.y;

    // the threads number in Block
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;
}