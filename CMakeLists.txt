#cmake_minimum_required(VERSION 3.26)
#project(KIP LANGUAGES CXX CUDA)  # Aggiungi sia CXX che CUDA

#set(CMAKE_CXX_STANDARD 17)

# Aggiungi i file sorgenti per il progetto
#set(SOURCE_FILES main.cpp main.cu
#        helloWorld.cu
#        helloWorld.cu
#        greeter.h)
#add_executable(KIP ${SOURCE_FILES}
#        helloWorld.cu)

# Impostazioni CUDA
# Abilita la compilazione separabile (utile per file di grandi dimensioni o progetti complessi)
#set_target_properties(KIP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_STANDARD 17)  # Standard per CUDA

# Abilita OpenMP
#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#    target_compile_options(KIP PRIVATE ${OpenMP_CXX_FLAGS})
#    target_link_libraries(KIP PRIVATE OpenMP::OpenMP_CXX)
#endif()

# Opzioni specifiche di compilazione per OpenMP
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#include_directories(/usr/local/include)  # Necessario solo per macOS
#link_directories(/usr/local/lib)         # Necessario solo per macOS

# Collega le librerie CUDA
#find_package(CUDA REQUIRED)
#target_link_libraries(KIP PRIVATE ${CUDA_LIBRARIES})

#cmake_minimum_required(VERSION 3.26)
#project(KIP LANGUAGES CXX CUDA)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 17)

# Aggiungi i file sorgenti per il progetto
#set(SOURCE_FILES main.cpp main.cu helloWorld.cu greeter.h)
#add_executable(KIP ${SOURCE_FILES})

# Impostazioni CUDA
# Abilita la compilazione separabile per CUDA
#set_target_properties(KIP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Configura le librerie CUDA
#find_package(CUDA REQUIRED)
#target_link_libraries(KIP PRIVATE CUDA::cudart)  # Usa CUDA::cudart per il runtime CUDA

# OpenMP
#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#target_compile_options(KIP PRIVATE ${OpenMP_CXX_FLAGS})
#target_link_libraries(KIP PRIVATE OpenMP::OpenMP_CXX)
#endif()

# Opzioni di compilazione specifiche per OpenMP
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Percorsi di librerie e inclusione per piattaforme specifiche
#if(APPLE)
#include_directories(/usr/local/include)
#link_directories(/usr/local/lib)
#endif()

# Imposta le directory di inclusione per CUDA (controlla su Windows e Unix)
#if(WIN32)
#target_include_directories(KIP PRIVATE "$ENV{CUDA_PATH}/include")
#elseif(UNIX)
#target_include_directories(KIP PRIVATE "/usr/local/cuda/include")
#endif()

cmake_minimum_required(VERSION 3.22)
project(KIP LANGUAGES CXX CUDA)

# Imposta gli standard di C++ e CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Aggiungi i file sorgenti per il progetto
set(SOURCE_FILES main.cpp helloWorld.cu)
add_executable(KIP ${SOURCE_FILES})

# Imposta le directory di inclusione e librerie CUDA
#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})
#link_directories(${CUDA_LIBRARIES_DIRS})

# Collega le librerie necessarie
target_link_libraries(KIP PRIVATE cudart) #${CUDA_LIBRARIES}
