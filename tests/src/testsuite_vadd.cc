/**
 * @file testsuite_vadd.cc
 * @author zhe.zhang
 * @date 2025-03-19 15:50:10
 * @brief Implement the test fixture class for vadd
 * @attention 
 */

#include "testsuite_vadd.h"

#include <iostream>

void VaddTestsuite::SetUpTestSuite(){
    std::cout << "=== Test Suite vadd Setup ===" << std::endl;
}

void VaddTestsuite::TearDownTestSuite(){
    std::cout << "=== Test Suite vadd Teardown ===" << std::endl;
}

void VaddTestsuite::SetUp(){
    std::cout << "[Test Setup]" << std::endl;
}

void VaddTestsuite::TearDown(){
    std::cout << "[Test Teardown]" << std::endl;
}