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

static bool check_paras(const float* const d_a, const float* const d_b, const float* const d_c, const int n){
   if((nullptr == d_a) || (nullptr == d_b) || (nullptr == d_c) || (0 >= n)){
      fprintf(stderr, "%s(%u): Invalid arguments: d_a=%p, d_b=%p, d_c=%p, n=%d\n", __FUNCTION__, __LINE__, d_a, d_b, d_c, n);
      fflush(stderr);
      return false;
   }

   cudaPointerAttributes attr_a, attr_b, attr_c;
   cudaPointerGetAttributes(&attr_a, d_a);
   cudaPointerGetAttributes(&attr_b, d_b);
   cudaPointerGetAttributes(&attr_c, d_c);
   if(!((cudaMemoryTypeDevice == attr_a.type) && (cudaMemoryTypeDevice == attr_b.type) && (cudaMemoryTypeDevice == attr_c.type))){
      fprintf(stderr, "%s(%u): Invalid memory type: attr_a.type=%d, attr_b.type=%d, attr_c.type=%d\n",
              __FUNCTION__, __LINE__, attr_a.type, attr_b.type, attr_c.type);
      fflush(stderr);
      return false;
   }

   return true;
}

int vadd(const float* d_a, const float* d_b, float* d_c, int n, cudaStream_t stream) {
   /// If considering performance overhead, comment out the parameter checking function
   if(!check_paras(d_a, d_b, d_c, n)){
      return -1;
   }
 
   launch_vadd(d_a, d_b, d_c, n, stream);

   return 0;
}