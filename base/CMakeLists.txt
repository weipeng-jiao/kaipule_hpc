cmake_minimum_required(VERSION 3.10)

project(main)

# set compile flags
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} " -std=c99 ")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -std=c++11  -fpermissive ")


find_package(OpenCL REQUIRED)

# 头文件路径
include_directories(${OpenCL_INCLUDE_DIR} 
                    ${CMAKE_SOURCE_DIR}/inc
                    ${CMAKE_SOURCE_DIR}/warpper/opencl 
                    )

file(GLOB DEMO_SRC ${CMAKE_SOURCE_DIR}/warpper/opencl/*.cpp
                    ${CMAKE_SOURCE_DIR}/warpper/opencl/*.c) 

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main05.cpp ${DEMO_SRC}) # use C/Cpp compiler

target_link_libraries(${PROJECT_NAME}  OpenCL::OpenCL)

            