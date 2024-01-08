# 生成NEON指令测试可执行文件
cmake_minimum_required(VERSION 3.10)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()


# 绝对路径
# set(OpenCV_DIR "/root/repo/sobel_grad_re/3rdparty/OpenCV/arm64-v8a/sdk/native/jni")
# message(STATUS "opencv dir:" ${OpenCV_DIR})

project(sobel_test)
# set compile flags
if(CMAKE_SYSTEM_NAME MATCHES "Android") # android arm
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -std=c99 -lm -llog")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC  -march=armv8.2-a+fp16  -std=c++11 -lm -llog")
else()
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} " -std=c99 ")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -std=c++11  -fpermissive ")
endif()

# 编译信息
message(STATUS "CMAKE_BUILD_TYPE :  ${CMAKE_BUILD_TYPE}")
message(STATUS "Runtime platform: ${CMAKE_SYSTEM_NAME}")

# 编译器路径打印
message(STATUS "C Complier: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Complier: ${CMAKE_CXX_COMPILER}")

# OpenCV
# find_package(OpenCV REQUIRED opencv_calib3d)

# 头文件路径
include_directories(  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/3rdparty/OpenCL ${CMAKE_SOURCE_DIR}/src)
# message(STATUS "opencv dir:" ${OpenCV_INCLUDE_DIRS})

# 加载源文件
set (SRC_LIST  ${CMAKE_SOURCE_DIR}/3rdparty/OpenCL/cl_library.cpp)

# 生成可执行文件
file(GLOB DEMO_SRC  ${CMAKE_SOURCE_DIR}/example/main02.cpp ${SRC_LIST})
add_executable(${PROJECT_NAME} ${DEMO_SRC})

target_link_libraries(${PROJECT_NAME} ${CMAKE_DL_LIBS})
# target_link_libraries (${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
file(GLOB bin ${CMAKE_SOURCE_DIR}/build/${PROJECT_NAME}) 
install(FILES  ${bin} DESTINATION example) 