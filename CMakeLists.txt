cmake_minimum_required(VERSION 3.0)
project(myCNN)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(main src/main.cpp ${SRC_FILES})

find_package(Eigen3 REQUIRED)
target_link_libraries(main Eigen3::Eigen)

if (DEFINED LOAD_JSON)
    set(LOAD_JSON ON)
endif()

if (DEFINED SAVE_JSON)
    set(SAVE_JSON ON)
endif()

if (DEFINED TEST)
    add_executable(load_file_test tests/load_file_test.cpp ${SRC_FILES})
    target_link_libraries(load_file_test Eigen3::Eigen)
endif()