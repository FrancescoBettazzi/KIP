#include <cuda_runtime.h>
#include <iostream>
//#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/stb_image_write.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            exit(err); \
        } \
    } while (0)

// CUDA kernel to invert colors
__global__ void invert_colors(unsigned char* d_image, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            d_image[idx + c] = 255 - d_image[idx + c];
        }
    }
}

void run_kernel(const char* inputPath, const char* outputPath) {
    int width, height, channels;

    // Load the image
    unsigned char* h_image = stbi_load(inputPath, &width, &height, &channels, 0);
    if (!h_image) {
        std::cerr << "Error loading image: " << inputPath << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t imageSize = width * height * channels;

    // Allocate memory on the device
    unsigned char* d_image;
    CUDA_CHECK(cudaMalloc(&d_image, imageSize));

    // Copy image data to device
    CUDA_CHECK(cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    invert_colors<<<gridSize, blockSize>>>(d_image, width, height, channels);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost));

    // Save the processed image
    if (!stbi_write_png(outputPath, width, height, channels, h_image, width * channels)) {
        std::cerr << "Error saving image: " << outputPath << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Image processed and saved to: " << outputPath << std::endl;

    // Free resources
    stbi_image_free(h_image);
    CUDA_CHECK(cudaFree(d_image));
}
