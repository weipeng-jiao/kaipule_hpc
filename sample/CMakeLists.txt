cmake_minimum_required(VERSION 3.10)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()


set(OpenCV_DIR "/home/jiao/repo/3rdparty/opencv/opencv-4.7.0/build")


project(sobelgrad)
# set compile flags

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} " -std=c99 ")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -std=c++11  -fpermissive ")


# OpenCV
find_package(OpenCV REQUIRED opencv_core opencv_highgui opencv_imgcodecs opencv_calib3d)
find_package(OpenCL REQUIRED)

# 头文件路径
include_directories(${CMAKE_SOURCE_DIR}/inc ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})

file(GLOB DEMO_SRC  ${CMAKE_SOURCE_DIR}/main.cpp)

add_executable(${PROJECT_NAME} ${DEMO_SRC}) # use C/Cpp compiler
# cuda_add_executable(${PROJECT_NAME} ${DEMO_SRC})  # use NVCC call C/Cpp compiler

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} OpenCL::OpenCL)

                
