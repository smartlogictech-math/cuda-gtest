/**
 * @file TEST_vadd.cu
 * @author zhe.zhang
 * @date 2025-03-18 16:20:18
 * @brief
 * @attention
 */

#include "vadd.h"
#include "testsuite_vadd.h"

#include <iostream>
#include "common.h"

TEST_F(VaddTestsuite, length_1024)
{
    const uint32_t N = 1024;
    float *h_A, *h_B, *h_C;

    CHECK(cudaMallocHost(&h_A, N * sizeof(float)));
    CHECK(cudaMallocHost(&h_B, N * sizeof(float)));
    CHECK(cudaMallocHost(&h_C, N * sizeof(float)));

    for (uint32_t i = 0; i < N; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t eStart, eEnd;
    cudaEventCreate(&eStart);
    cudaEventCreate(&eEnd);
    cudaEventRecord(eStart);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(eEnd);
    cudaEventSynchronize(eEnd);
    float time;
    cudaEventElapsedTime(&time, eStart, eEnd);
    std::cout << "Elapesd time: " << time << "ms" << std::endl;

    CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证结果
    bool correct = true;
    for (uint32_t i = 0; i < N; i++)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            correct = false;
            std::cout << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            break;
        }
    }

    if (correct)
    {
        std::cout << "Vector addition successful!" << std::endl;
    }

    CHECK(cudaEventDestroy(eStart));
    CHECK(cudaEventDestroy(eEnd));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(h_C));

    ASSERT_TRUE(correct);
}