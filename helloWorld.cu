//
// Created by Francesco on 14/11/2024.
//
#include <cstdio>               // Per printf
#include <cuda_runtime.h>       // Per funzioni CUDA come cudaDeviceReset
#include <string>

// Kernel CUDA
__global__ void helloFromGPU() {
    printf("Hello world from GPU thread %d!\n", threadIdx.x);
}

__global__ void testKernel() {
    printf("Testing printf from GPU thread %d\n", threadIdx.x);
}


void greet(const std::string& name) {
    printf("Hello %s!\n", name.c_str());
}

// Funzione main
void runKernel() {
    // Saluto dall'host
    greet("Pinco"); // Usa una C-string al posto di std::string
    testKernel<<<1, 10>>>();

    // Lancio del kernel CUDA
    helloFromGPU<<<1, 10>>>();

    // Controlla eventuali errori nel lancio del kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Sincronizza il dispositivo per aspettare il completamento di tutti i thread
    cudaDeviceSynchronize();

    // Reset del dispositivo CUDA
    cudaDeviceReset();
}


// Funzione main
/*int runKernel() {
    // Saluto dall'host
    greet("Pinco"); // Usa una C-string al posto di std::string
    testKernel<<<1, 10>>>();

    // Lancio del kernel CUDA
    helloFromGPU<<<1, 10>>>();

    // Controlla eventuali errori nel lancio del kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Sincronizza il dispositivo per aspettare il completamento di tutti i thread
    cudaDeviceSynchronize();

    // Reset del dispositivo CUDA
    cudaDeviceReset();

    return 0;
}*/
