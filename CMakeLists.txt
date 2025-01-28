cmake_minimum_required(VERSION 3.22)
project(KIP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(SOURCE_FILES main.cpp main.cu)
add_executable(KIP ${SOURCE_FILES})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES_DIRS})

target_link_libraries(KIP PRIVATE cudart) #${CUDA_LIBRARIES}
