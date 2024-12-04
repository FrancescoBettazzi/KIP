#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void applyKernelCUDA(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfSize = kernelSize / 2;

    if (x >= halfSize && y >= halfSize && x < width - halfSize && y < height - halfSize) {
        float accum[3] = {0.0f, 0.0f, 0.0f};

        // Iterazione sul kernel
        for (int ky = -halfSize; ky <= halfSize; ky++) {
            for (int kx = -halfSize; kx <= halfSize; kx++) {
                int px = (y + ky) * width + (x + kx);
                for (int c = 0; c < channels; c++) {
                    accum[c] += inputImage[px * channels + c] * kernel[(ky + halfSize) * kernelSize + (kx + halfSize)];
                }
            }
        }

        int outIdx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            outputImage[outIdx + c] = min(max(int(accum[c]), 0), 255);
        }
    }
}

int main() {
    int width = 1024;  // Imposta larghezza immagine
    int height = 1024; // Imposta altezza immagine
    int channels = 3;  // Canali colore

    // Alloca immagine di input e output su host
    unsigned char* h_image = ... // Carica l'immagine come hai fatto finora
    unsigned char* h_outputImage = new unsigned char[width * height * channels];

    // Alloca memoria su device
    unsigned char *d_image, *d_outputImage;
    float *d_kernel;
    cudaMalloc(&d_image, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_outputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_kernel, 9 * sizeof(float));

    // Carica kernel (es. un kernel 3x3, ma puoi adattare)
    float h_kernel[9] = { ... }; // Inserisci il kernel che desideri

    // Copia dati host->device
    cudaMemcpy(d_image, h_image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // Impostazione grid e blocchi
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Lancio kernel
    applyKernelCUDA<<<gridSize, blockSize>>>(d_image, d_outputImage, width, height, channels, d_kernel, 3);

    // Copia risultato device->host
    cudaMemcpy(h_outputImage, d_outputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Libera memoria
    cudaFree(d_image);
    cudaFree(d_outputImage);
    cudaFree(d_kernel);
    delete[] h_outputImage;
    delete[] h_image;

    return 0;
}
