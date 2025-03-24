/**
 * @file vadd_api.h
 * @author zhe.zhang
 * @date 2025-03-23 17:14:34
 * @brief 
 * @attention 
 */
#pragma once

#include <cuda_runtime.h>

/**
 * @brief vector add, c=a+b
 * 
 * @param d_a address of a in device memory
 * @param d_b address of b in device memory
 * @param d_c address of c in device memory
 * @param n length of vector
 * @param stream 
 * @return int 
 * @retval 0: success
 * @retval -1: failure
 */
int vadd(const float* d_a, const float* d_b, float* d_c, int n, cudaStream_t stream);