cmake_minimum_required(VERSION 2.8.3)
project(unitree_legged_sdk)

include_directories(include)

link_directories(lib)

add_compile_options(-std=c++11)

set(EXTRA_LIBS -pthread libunitree_legged_sdk_amd64.so lcm)

set(CMAKE_CXX_FLAGS "-O3")

add_subdirectory(pybind11)
pybind11_add_module(robot_interface python_interface.cpp)
target_link_libraries(robot_interface ${EXTRA_LIBS})
