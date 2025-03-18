cmake_minimum_required(VERSION 3.22.1)
project(empty_project LANGUAGES C CXX CUDA)

option(BUILD_LIB "Build library only" OFF)
option(BUILD_DEBUG_TESTS "Build debug tests" OFF)
option(BUILD_RELEASE_TESTS "Build release tests" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(BUILD_LIB ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug_test")
    set(BUILD_DEBUG_TESTS ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release_test")
    set(BUILD_RELEASE_TESTS ON)
endif()

set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

file(GLOB_RECURSE SRC_FILES "${SRC_DIR}/*.c" "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.cu" "${SRC_DIR}/*.sc")
message(STATUS "SRC_FILES: ${SRC_FILES}")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/add)

# build library
if(BUILD_LIB)
    add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
    target_compile_options(${PROJECT_NAME} PRIVATE
        -O2
        -g
        -Wall
        -Werror
    )
    message(STATUS "Building library with src files.")
endif()

# Debug
if(BUILD_DEBUG_TESTS)
    set(GTEST_DIR /home/zhe.zhang/Public/lib/googletest)
    set(TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/tests)

    file(GLOB_RECURSE TEST_FILES "${TEST_DIR}/*.c" "${TEST_DIR}/*.cpp" "${TEST_DIR}/*.cu" "${TEST_DIR}/*.sc")
    message(STATUS "TEST_FILES: ${TEST_FILES}")
    
    include_directories(${GTEST_DIR}/include)

    link_directories(${GTEST_DIR}/lib)

    add_executable(test_elf ${SRC_FILES} ${TEST_FILES})
    target_link_libraries(test_elf pthread libgtest.a)
    set_target_properties(test_elf PROPERTIES OUTPUT_NAME "test_${PROJECT_NAME}.elf")
    target_compile_options(test_elf PRIVATE
        -O0
        -g
        -Wall
        -Werror
    )
    message(STATUS "Building tests with src and test files.")
endif()

# Release
if(BUILD_RELEASE_TESTS)
    set(GTEST_DIR /home/zhe.zhang/Public/lib/googletest)
    set(TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/tests)

    file(GLOB_RECURSE TEST_FILES "${TEST_DIR}/*.c" "${TEST_DIR}/*.cpp" "${TEST_DIR}/*.cu" "${TEST_DIR}/*.sc")
    message(STATUS "TEST_FILES: ${TEST_FILES}")
    
    include_directories(${GTEST_DIR}/include)

    link_directories(${GTEST_DIR}/lib)

    add_executable(test_elf ${SRC_FILES} ${TEST_FILES})
    target_link_libraries(test_elf pthread libgtest.a)
    set_target_properties(test_elf PROPERTIES OUTPUT_NAME "test_${PROJECT_NAME}.elf")
    target_compile_options(test_elf PRIVATE
        -O2
        -g
        -Wall
        -Werror
    )
    message(STATUS "Building tests with src and test files.")
endif()