cmake_minimum_required(VERSION 3.0)
project(myCNN)

file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_executable(main ${SRC_FILES})

include_directories(${CMAKE_SOURCE_DIR}/include)