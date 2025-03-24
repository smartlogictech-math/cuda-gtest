/**
 * @file vadd_demo.cc
 * @author zhe.zhang
 * @date 2025-03-24 11:29:58
 * @brief
 * @attention
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "vadd_api.h"

int main()
{
  const int N = 1024;
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMemcpyAsync(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);

  int ret = vadd(d_a, d_b, d_c, N, stream);
  if (0 != ret)
  {
    printf("Error: vadd return %d\n", ret);
  }
  else
  {
    cudaMemcpyAsync(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("Info: vadd completed\n");
  }

  cudaStreamDestroy(stream);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}