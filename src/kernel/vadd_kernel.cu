/**
 * @file vadd_kernel.cu
 * @author zhe.zhang
 * @date 2025-03-23 17:33:29
 * @brief 
 * @attention 
 */

#include <cuda_runtime.h>

__global__ void vadd_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vadd(const float* d_a, const float* d_b, float* d_c, int n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vadd_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
}