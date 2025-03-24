/**
 * @file TEST_vadd_kernel.cu
 * @author zhe.zhang
 * @date 2025-03-23 17:48:17
 * @brief 
 * @attention 
 */

#include "internal/kernel/vadd_kernel.h"

#include <gtest/gtest.h>

TEST(VaddKernelTest, BasicAddition) {
  const int n = 3;
  float h_a[3] = {1.0f, 2.0f, 3.0f};
  float h_b[3] = {4.0f, 5.0f, 6.0f};
  float h_c[3] = {0};

  float *d_a, *d_b, *d_c;
  cudaError_t err;

  err = cudaMalloc(&d_a, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMalloc(&d_b, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMalloc(&d_c, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  err = cudaMemcpyAsync(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMemcpyAsync(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);

  launch_vadd(d_a, d_b, d_c, n, 0);

  err = cudaMemcpyAsync(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  ASSERT_EQ(err, cudaSuccess);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  EXPECT_FLOAT_EQ(h_c[0], 5.0f);
  EXPECT_FLOAT_EQ(h_c[1], 7.0f);
  EXPECT_FLOAT_EQ(h_c[2], 9.0f);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

TEST(VaddKernelTest, LargeScaleAddition) {
  const int n = 1 << 20;
  std::vector<float> h_a(n, 1.0f);
  std::vector<float> h_b(n, 2.0f);
  std::vector<float> h_c(n, 0.0f);

  float *d_a, *d_b, *d_c;
  cudaError_t err;

  err = cudaMalloc(&d_a, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMalloc(&d_b, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMalloc(&d_c, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  err = cudaMemcpyAsync(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMemcpyAsync(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);

  launch_vadd(d_a, d_b, d_c, n, stream);

  err = cudaMemcpyAsync(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  ASSERT_EQ(err, cudaSuccess);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(h_c[i], 3.0f, 1e-5);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
