cmake_minimum_required(VERSION 3.14)
project(main)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)

cuda_add_executable(${PROJECT_NAME} main.cu)
