#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main () {
    int m = 4096, n = 4096, k = 4096;
    printf("shape: (%d %d) x (%d %d)\n", m ,k, k, n);
    int start_algo = CUBLAS_GEMM_DEFAULT;
    
}