cmake_minimum_required(VERSION 3.10)
project(sobelgrad)

if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]" OR CMAKE_C_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif ()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED core highgui imgproc features2d)
find_package(OpenCL REQUIRED)

# 头文件路径
include_directories(${CMAKE_SOURCE_DIR}/inc ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})

file(GLOB DEMO_SRC  ${CMAKE_SOURCE_DIR}/main.cu)

# add_executable(${PROJECT_NAME} ${DEMO_SRC}) # use C/Cpp compiler
cuda_add_executable(${PROJECT_NAME} ${DEMO_SRC})  # use NVCC call C/Cpp compiler

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} OpenCL::OpenCL)

                
