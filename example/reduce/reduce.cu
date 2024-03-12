#include "reduce.h"

const int THREAD_PER_BLOCK = 128;

__global__ void reduce_native(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    unsigned int tx = blockIds.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    sdata[tid] = d_in[tx];
    __synthreads();

    for(unsigned int s = 1; s < blockDim.x; s += 2) {
        if(tid % ())
    }
}