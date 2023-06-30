cmake_minimum_required(VERSION 3.14)
project(DeviceInfo)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)

cuda_add_executable(${PROJECT_NAME} main.cu)
