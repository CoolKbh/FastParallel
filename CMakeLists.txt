cmake_minimum_required(VERSION 3.10)

PROJECT(fastparallel)

add_subdirectory(${PROJECT_SOURCE_DIR}/fastparallel)
add_subdirectory(${PROJECT_SOURCE_DIR}/example)

##################################### Building: FastParallel C++ SDK #######################################
file(GLOB_RECURSE ALL_PARALLEL_SRCS ${PROJECT_SOURCE_DIR}/fastparallel)