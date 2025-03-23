/**
 * @file vadd_api.h
 * @author zhe.zhang
 * @date 2025-03-23 17:14:34
 * @brief 
 * @attention 
 */
#pragma once

int vadd(const float* a, const float* b, float* c, int n, cudaStream_t stream);