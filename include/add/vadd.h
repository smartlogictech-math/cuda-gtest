/**
 * @file vadd.h
 * @author zhe.zhang
 * @date 2025-03-18 16:16:42
 * @brief 
 * @attention 
 */
#ifndef _VADD_H_
#define _VADD_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

    __global__ void vectorAdd(const float *A, const float *B, float *C, int n);

#ifdef __cplusplus
}
#endif

#endif // _VADD_H_