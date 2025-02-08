#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <tuple>

using namespace std;
using KernelType = vector<vector<float>>;

// Kernel CUDA
__global__
void doConvolutionCUDA(const unsigned char *inputImage, unsigned char *outputImage,
                       int width, int height, int channels, const float *kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Coordinata X del thread
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Coordinata Y del thread

    int halfSize = kernelSize / 2;

    if (x >= halfSize && x < width - halfSize && y >= halfSize && y < height - halfSize) {
        float accumulators[3] = {0.0f, 0.0f, 0.0f};

        for (int ky = -halfSize; ky <= halfSize; ky++) {
            for (int kx = -halfSize; kx <= halfSize; kx++) {
                int pixelPos = ((y + ky) * width + (x + kx)) * channels;

                for (int c = 0; c < channels; c++) {
                    accumulators[c] += inputImage[pixelPos + c] *
                                       kernel[(ky + halfSize) * kernelSize + (kx + halfSize)];
                }
            }
        }

        int outPos = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            outputImage[outPos + c] = min(max(int(accumulators[c]), 0), 255);
        }
    }
}

tuple<vector<unsigned char>, double> applyKernelCUDA(
        const unsigned char *image, int width, int height,
        const KernelType &kernel, int channels) {

    auto startTime = chrono::high_resolution_clock::now();

    int kernelSize = kernel.size();
    int imageSize = width * height * channels;
    int kernelElements = kernelSize * kernelSize;

    // Convertire il kernel in un array lineare
    vector<float> flatKernel(kernelElements);
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            flatKernel[i * kernelSize + j] = kernel[i][j];
        }
    }

    // Allocare memoria sull'host
    vector<unsigned char> outputImage(imageSize);

    // Allocare memoria sul device
    unsigned char *d_inputImage, *d_outputImage;
    float *d_kernel;

    cudaMalloc(&d_inputImage, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_outputImage, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_kernel, kernelElements * sizeof(float));

    // Copiare i dati dall'host al device
    cudaMemcpy(d_inputImage, image, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, flatKernel.data(), kernelElements * sizeof(float), cudaMemcpyHostToDevice);

    // Definire dimensione dei blocchi e della griglia
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Lanciare il kernel CUDA
    doConvolutionCUDA<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, channels, d_kernel,
                                               kernelSize);

    // Copiare i risultati dal device all'host
    cudaMemcpy(outputImage.data(), d_outputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Deallocare memoria sul device
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_kernel);

    auto endTime = chrono::high_resolution_clock::now();
    double executionTime = chrono::duration<double>(endTime - startTime).count();

    return make_tuple(outputImage, executionTime);
}
