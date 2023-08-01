cmake_minimum_required(VERSION 3.10)

project(main)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -std=c++17 ")

find_package(Threads) 
add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main02.cpp)
target_link_libraries (${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT}) 