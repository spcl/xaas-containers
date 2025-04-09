cmake_minimum_required(VERSION 3.13)
project(omp-finder)

find_package(Clang REQUIRED CONFIG)

include_directories(${CLANG_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(omp-finder omp_finder.cpp)

target_link_libraries(omp-finder
  clangTooling
  clangFrontend
  clangASTMatchers
  clangAST
  clangBasic
)
