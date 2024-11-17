cmake_minimum_required(VERSION 3.0)
project(myCNN)

file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_executable(main ${SRC_FILES})

find_package(Eigen3 REQUIRED)
target_link_libraries(main Eigen3::Eigen)

include_directories(${CMAKE_SOURCE_DIR}/include)