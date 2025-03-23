/**
 * @file TEST_vadd_api.cu
 * @author zhe.zhang
 * @date 2025-03-23 21:15:16
 * @brief 
 * @attention 
 */

#include "vadd_api.h"

#include "fixtures/vadd_api_fixture.h"

#include <gtest/gtest.h>

TEST_F(VaddAPITest, SmallScaleTest) {
    const int n = 5;
    float h_a[n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_b[n] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float h_c[n] = {0};

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpyAsync(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice, stream);

    vadd(d_a, d_b, d_c, n, stream);
    cudaMemcpyAsync(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    EXPECT_FLOAT_EQ(h_c[0], 6.0f);
    EXPECT_FLOAT_EQ(h_c[1], 6.0f);
    EXPECT_FLOAT_EQ(h_c[2], 6.0f);
    EXPECT_FLOAT_EQ(h_c[3], 6.0f);
    EXPECT_FLOAT_EQ(h_c[4], 6.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST_F(VaddAPITest, LargeScaleTest) {
    const int n = 1 << 20;
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpyAsync(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);

    vadd(d_a, d_b, d_c, n, stream);
    cudaMemcpyAsync(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < n; i += n / 100) {
        EXPECT_NEAR(h_c[i], 3.0f, 1e-5);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST_F(VaddAPITest, InvalidInput) {
    float *d_c;
    cudaMalloc(&d_c, sizeof(float));

    EXPECT_EQ(vadd(nullptr, nullptr, d_c, 1, stream), -1);
    EXPECT_EQ(vadd(nullptr, nullptr, nullptr, 1, stream), -1);
    EXPECT_EQ(vadd(nullptr, nullptr, nullptr, -1, stream), -1);

    cudaFree(d_c);
}
