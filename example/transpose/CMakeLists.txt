cmake_minimum_required(VERSION 3.10)

project(cuda_gemm_demo LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 52)
find_package(CUDA REQUIRED) 
include_directories("${CUDA_INCLUDE_DIRS}") 

include_directories(${CMAKE_SOURCE_DIR})

file(GLOB CUDA_SRC ${CMAKE_SOURCE_DIR}/cuda_gemm.cu)
add_library(matrix ${CUDA_SRC})

add_executable(main main.cc)
target_link_libraries(main matrix)
target_link_libraries(main  ${CUDA_cublas_LIBRARY})