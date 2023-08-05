cmake_minimum_required(VERSION 3.10)

project(eigen_gemm_demo)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(eigen_gemm eigen_gemm.cc)