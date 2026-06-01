cmake_minimum_required(VERSION 3.13)
project(omp-finder)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(${CLANG_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(omp-finder omp_finder.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

target_link_libraries(omp-finder
  #linking against these means that the final executable will link against the shared libraries
  clang-cpp
  LLVM

  #full enumeration of all the clang components we're actually compiling against
  # clangTooling
  # clangFrontend
  # clangASTMatchers
  # clangAST
  # clangBasic
  # clangSerialization
  # LLVMSupport
)
