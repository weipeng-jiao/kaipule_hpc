cmake_minimum_required(VERSION 3.10)

set (CMAKE_BUILD_TYPE Debug)

project(main)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O0 -mavx2 -std=c++17 ")

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main01.cpp)
