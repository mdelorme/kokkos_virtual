cmake_minimum_required(VERSION 3.16)

project(kokkos_virtual LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external/kokkos)

add_executable(test1 test1.cpp)
target_link_libraries(test1 Kokkos::kokkos)
