cmake_minimum_required(VERSION 3.22.1)
project(vadd LANGUAGES C CXX CUDA)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -Xcompiler=-Wall")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -G -g -Xcompiler=-Wall")
endif()

set(SRC_FILE_EXTENSIONS "*.c" "*.cpp" "*.cc" "*.cu" "*.sc")

set(project_lib_name ${PROJECT_NAME})
add_subdirectory(src)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

option(BUILD_SAMPLES "Build samples" ON)
if(BUILD_SAMPLES)
  add_subdirectory(samples)
endif()