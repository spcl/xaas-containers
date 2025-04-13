cmake_minimum_required(VERSION 3.13.4)
project(ReplaceTargetFeatures)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

link_directories(${LLVM_LIBRARY_DIRS})

add_library(ReplaceTargetFeatures MODULE features-pass.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

llvm_map_components_to_libnames(llvm_libs Core Passes Support)

target_link_libraries(ReplaceTargetFeatures PRIVATE ${llvm_libs})
