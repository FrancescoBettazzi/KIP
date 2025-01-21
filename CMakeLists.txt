cmake_minimum_required(VERSION 3.22)
project(KIP LANGUAGES CXX CUDA)

# Imposta gli standard di C++ e CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Aggiungi i file sorgenti per il progetto
set(SOURCE_FILES main.cpp main.cu)
add_executable(KIP ${SOURCE_FILES})

# Imposta le directory di inclusione e librerie CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES_DIRS})

# Collega le librerie necessarie
target_link_libraries(KIP PRIVATE cudart) #${CUDA_LIBRARIES}
