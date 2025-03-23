/**
 * @file testsuite_vadd.h
 * @author zhe.zhang
 * @date 2025-03-19 16:01:40
 * @brief Define a test fixture class for vadd
 * @attention 
 */
#ifndef _TESTSUITE_VADD_H_
#define _TESTSUITE_VADD_H_

#include <gtest/gtest.h>

// Define a test fixture class
class VaddTestsuite : public ::testing::Test {
    protected:
        // Runs once before any test in this test suite
        static void SetUpTestSuite();
        // Runs once after all tests in this test suite have run
        static void TearDownTestSuite();

        // Runs before each test case
        void SetUp() override;
        // Runs after each test case
        void TearDown() override;
};

#endif // _TESTSUITE_VADD_H_