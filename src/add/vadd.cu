/**
 * @file vadd.cu
 * @author zhe.zhang
 * @date 2025-03-18 16:15:38
 * @brief 
 * @attention 
 */

#include "vadd.h"

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}