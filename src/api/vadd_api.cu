/**
 * @file vadd_api.c
 * @author zhe.zhang
 * @date 2025-03-23 17:34:31
 * @brief 
 * @attention 
 */

#include "vadd_api.h"
#include "internal/kernel/vadd_kernel.h"

#include <cstdio>

static bool check_paras(const float* a, const float* b, float* c, int n){
   return !((nullptr == a) || (nullptr == b) || (nullptr == c) || (0 >= n));
}

int vadd(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
   if(!check_paras(a, b, c, n)){
      fprintf(stderr, "Invalid arguments in vadd: a=%p, b=%p, c=%p, n=%d\n", a, b, c, n);
         fflush(stderr);
         return -1;
   }
   float *d_a, *d_b, *d_c;

   cudaMalloc(&d_a, n * sizeof(float));
   cudaMalloc(&d_b, n * sizeof(float));
   cudaMalloc(&d_c, n * sizeof(float));

   cudaMemcpyAsync(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice, stream);
   cudaMemcpyAsync(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice, stream);
 
   launch_vadd(d_a, d_b, d_c, n, stream);

   cudaMemcpyAsync(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream);

   cudaStreamSynchronize(stream);
 
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

   return 0;
}