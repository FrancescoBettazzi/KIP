//
// Created by Francesco on 04/11/2024.
//
#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <chrono>
#include <omp.h>
#include <tuple>
#include <iomanip>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

using namespace std;
using KernelType = vector<vector<float>>;

const KernelType identity = {
                                {0, 0, 0},
                                {0, 1, 0},
                                {0, 0, 0}
                            };

const KernelType edgeDetection = {
                                      {0, -1, 0},
                                      {-1, 4, -1},
                                      {0, -1, 0}
                                  };

const KernelType edgeDetection2 = {
                                      {-1, -1, -1},
                                      {-1, 8, -1},
                                      {-1, -1, -1}
                                  };

const KernelType sharpen = {
                               {0, -1, 0},
                               {-1, 5, -1},
                               {0, -1, 0}
                           };

const KernelType boxBlur = {
                                {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
                                {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
                                {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}
                            };

const KernelType gaussianBlur = {
                                    {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
                                    {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
                                    {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
                                };

const KernelType gaussianBlur5X5 = {
                                         {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f},
                                         {4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
                                         {6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f},
                                         {4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
                                         {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f}
                                     };

const KernelType unsharpMasking5X5 = {
                                             {-1 / 256.0f, -4 / 256.0f, -6 / 256.0f, - 4 / 256.0f, -1 / 256.0f},
                                             {-4 / 256.0f, -16 / 256.0f, -24 / 256.0f, - 16 / 256.0f, -4 / 256.0f},
                                             {-6 / 256.0f, -24 / 256.0f, 476 / 256.0f, -24 / 256.0f, -6 / 256.0f},
                                             {-4 / 256.0f, -16 / 256.0f, -24 / 256.0f, -16 / 256.0f, -4 / 256.0f},
                                             {-1 / 256.0f, -4 / 256.0f, -6 / 256.0f, -4 / 256.0f, -1 / 256.0f}
                                         };

// Mappa per associare i nomi ai kernel
map<string, KernelType> kernels = {
        {"identity",            identity},
        {"edgeDetection",       edgeDetection},
        {"edgeDetection2",      edgeDetection2},
        {"sharpen",             sharpen},
        {"boxBlur",             boxBlur},
        {"gaussianBlur",        gaussianBlur},
        {"gaussianBlur5X5",   gaussianBlur5X5},
        {"unsharpMasking5X5", unsharpMasking5X5}
};


void printExecutionTimes(const map<string, vector<double>>& execution_times) {
    cout << endl;
    cout << left
         << setw(20) << ""
         << setw(20) << "Sequential (s)"
         << setw(20) << "OMP (s)"
         << setw(20) << "CUDA (s)"
         << endl;
    cout << string(80, '-') << endl;

    for (const auto& entry : execution_times) {
        cout << left
             << setw(20) << fixed << setprecision(6) << entry.first
             << setw(20) << fixed << setprecision(6) << entry.second[0]
             << setw(20) << fixed << setprecision(6) << entry.second[1]
             << setw(20) << fixed << setprecision(6) << 0.0//entry.second[2]
             << endl;
    }
}

auto applyKernel(
        const unsigned char *image, int width, int height,
        const KernelType &kernel, int channels) {

    auto start_time = chrono::high_resolution_clock::now();

    vector<unsigned char> outputImage(width * height * channels);
    int kernelSize = kernel.size();
    int halfSize = kernelSize / 2;


/** PSEUDOCODE
    for each image row in input image:
    for each pixel in image row:

    set accumulator to zero

    for each kernel row in kernel:
    for each element in kernel row:

    if element position  corresponding* to pixel position then
    multiply element value  corresponding* to pixel value
    add result to accumulator
    endif

    set output image pixel to accumulator
*/

    for (int y = halfSize; y < height - halfSize; y++) {
        for (int x = halfSize; x < width - halfSize; x++) {
            float accumulators[3] = {0.0f, 0.0f, 0.0f};

            for (int ky = -halfSize; ky <= halfSize; ky++) {
                for (int kx = -halfSize; kx <= halfSize; kx++) {
                    int pixelPos = (y + ky) * width * channels + (x + kx) * channels;
                    for (int c = 0; c < channels; c++) {
                        accumulators[c] += image[pixelPos + c] * kernel[ky + halfSize][kx + halfSize];
                    }
                }
            }

            int outPos = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                outputImage[outPos + c] = min(max(int(accumulators[c]), 0), 255);
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    double execution_time = chrono::duration<double>(end_time - start_time).count();
    return make_tuple(outputImage, execution_time);
}

auto applyKernelOMP(
        const unsigned char *image, int width, int height,
        const KernelType &kernel, int channels) {

    auto start_time = chrono::high_resolution_clock::now();

    vector<unsigned char> outputImage(width * height * channels);
    int kernelSize = kernel.size();
    int halfSize = kernelSize / 2;

    #pragma omp parallel for collapse(2) schedule(guided)
    for (int y = halfSize; y < height - halfSize; y++) {
        for (int x = halfSize; x < width - halfSize; x++) {
            float accumulators[3] = {0.0f, 0.0f, 0.0f};

            for (int ky = -halfSize; ky <= halfSize; ky++) {
                for (int kx = -halfSize; kx <= halfSize; kx++) {
                    int pixelPos = (y + ky) * width * channels + (x + kx) * channels;
                    for (int c = 0; c < channels; c++) {
                        accumulators[c] += image[pixelPos + c] * kernel[ky + halfSize][kx + halfSize];
                    }
                }
            }

            int outPos = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                outputImage[outPos + c] = min(max(int(accumulators[c]), 0), 255);
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    double execution_time = chrono::duration<double>(end_time - start_time).count();
    return make_tuple(outputImage, execution_time);
}

// Dichiarazione della funzione CUDA definita in main.cu
void applyKernelCUDA(const char* inputPath, const char* outputPath);

int main() {

    map<string, vector<double>> execution_times;

    int width, height, channels;

    // TEST IMAGES
    //const char *path_img = "../img/test.png";
    //const char *path_img = "../img/panda.jpg";
    //const char *path_img = "../img/sheep.jpg";

    // LOCAL PATH
    const char *path_img = "../img/sheep.jpg";
    const char *output_path = "../img/results/";

    // SERVER DINFO PATH
    //const char *path_img = "img/sheep.jpg";
    //const char *output_path = "img/results/";


    if (!filesystem::exists(output_path))
        filesystem::create_directories(output_path);

    //req_comp 0=auto, 1=gray scale, 3=RGB
    int req_comp = 0;

    unsigned char *image = stbi_load(path_img, &width, &height, &channels, req_comp);
    if (!image) {
        cerr << "Error: can't open image!" << endl;
        return -1;
    }

    cout << "Computing Sequential Kernel Image Processing ..." << endl;

    for (const auto &[name, kernel] : kernels) {
        auto [outputImage, execution_time] = applyKernel(image, width, height, kernel, channels);
        stbi_write_jpg((output_path + name + ".jpg").c_str(), width, height, channels, outputImage.data(), 100);
        execution_times[name].emplace_back(execution_time);
    }

    cout << "Computing Parallel Kernel Image Processing ..." << endl;

    for (const auto &[name, kernel] : kernels) {
            auto [outputImage, execution_time] = applyKernelOMP(image, width, height, kernel, channels);
            stbi_write_jpg((output_path + name + "_parallel.jpg").c_str(), width, height, channels, outputImage.data(), 100);
            execution_times[name].emplace_back(execution_time);
    }

    stbi_image_free(image);

    printExecutionTimes(execution_times);

    //applyKernelCUDA(path_img, output_path);

    return 0;
}
