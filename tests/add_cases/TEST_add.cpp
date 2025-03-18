/**
 * @file TEST_add.cpp
 * @author zhe.zhang
 * @date 2025-03-18 15:12:44
 * @brief 
 * @attention 
 */
#include "add.h"

#include <gtest/gtest.h>

TEST(Add, OnePlusTwo){
    int val = add(1, 2);
    ASSERT_EQ(val, 3);
}