cmake_minimum_required(VERSION 3.22.1)
project(empty_project LANGUAGES C CXX CUDA)

option(BUILD_LIB "Build library only" OFF)
option(BUILD_DEBUG_TESTS "Build debug tests" OFF)
option(BUILD_RELEASE_TESTS "Build release tests" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(BUILD_LIB ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug_test")
    set(BUILD_DEBUG_TESTS ON)
endif()

set(SRC_FILE_EXTENSIONS "*.c" "*.cpp" "*.cc" "*.cu" "*.sc")

set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

set(SRC_FILES)
foreach(ext IN LISTS SRC_FILE_EXTENSIONS)
    file(GLOB_RECURSE tmp_files "${SRC_DIR}/${ext}")
    list(APPEND SRC_FILES ${tmp_files})
endforeach()
message(STATUS "SRC_FILES: ${SRC_FILES}")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/add)
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/common)

# build library
if(BUILD_LIB)
    add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
    target_compile_options(${PROJECT_NAME} PRIVATE
        -O2
        -g
        -Wall
    )
    message(STATUS "Building library with src files.")
endif()

# Debug
if(BUILD_DEBUG_TESTS)
    set(GTEST_DIR /usr/local)
    set(TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/tests)

    set(TEST_FILES)
    foreach(ext IN LISTS SRC_FILE_EXTENSIONS)
        file(GLOB_RECURSE tmp_files "${TEST_DIR}/${ext}")
        list(APPEND TEST_FILES ${tmp_files})
    endforeach()
    message(STATUS "TEST_FILES: ${TEST_FILES}")
    
    include_directories(${GTEST_DIR}/include)
    include_directories(${TEST_DIR}/inc)

    add_executable(test_elf ${SRC_FILES} ${TEST_FILES})

    link_directories(${GTEST_DIR}/lib)
    target_link_libraries(test_elf pthread libgtest.a)
    
    set_target_properties(test_elf PROPERTIES OUTPUT_NAME "test_${PROJECT_NAME}.elf")
    
    target_compile_options(test_elf PRIVATE
        -O2
        -g
        -Wall
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

    message(STATUS "Building tests with src and test files.")
endif()