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

const KernelType kernel = {
        {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
        {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
        {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}
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
             << setw(20) << fixed << setprecision(10) << entry.first
             << setw(20) << fixed << setprecision(10) << entry.second[0]
             << setw(20) << fixed << setprecision(10) << entry.second[1]
             << setw(20) << fixed << setprecision(10) << entry.second[2]
             << endl;
    }
}

string getFileName(const std::string& path) {
    size_t lastSlashPos = path.find_last_of("/\\");
    size_t lastDotPos = path.find_last_of(".");

    std::string fileName = (lastSlashPos == std::string::npos) ? path : path.substr(lastSlashPos + 1);

    if (lastDotPos != std::string::npos) {
        fileName = fileName.substr(0, lastDotPos);
    }

    return fileName;
}

tuple<vector<unsigned char>, double> applyKernel(
        const unsigned char *image, int width, int height,
        const std::vector<std::vector<float>> &kernel, int channels) {

    auto startTime = chrono::high_resolution_clock::now();

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

    auto endTime = chrono::high_resolution_clock::now();
    double executionTime = chrono::duration<double>(endTime - startTime).count();
    return make_tuple(outputImage, executionTime);
}

tuple<vector<unsigned char>, double> applyKernelOMP(
        const unsigned char *image, int width, int height,
        const KernelType &kernel, int channels) {

    auto startTime = chrono::high_resolution_clock::now();

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

    auto endTime = chrono::high_resolution_clock::now();
    double executionTime = chrono::duration<double>(endTime - startTime).count();
    return make_tuple(outputImage, executionTime);
}

// Dichiarazione della funzione CUDA definita in main.cu
tuple<vector<unsigned char>, double> applyKernelCUDA(
        const unsigned char *image, int width, int height,
        const KernelType &kernel, int channels);

int main() {

    map<string, vector<double>> executionTimes;

    // TEST IMAGES
    //const char *pathImg = "../img/test.png";
    //const char *pathImg = "../img/panda.jpg";
    //const char *pathImg = "../img/sheep.jpg";

    // LOCAL PATHS
    //const char *pathImg = "../img/sheep.jpg";
    const char *outputPath = "../img/results/";
    vector<string> pathImages = {
            "../img/01.280x180.jpg",
            "../img/02.299x160.jpg",
            "../img/03.480p.jpg",
            "../img/04.720p.jpg",
            "../img/05.1080p.jpg",
            "../img/06.4K.jpg",
            "../img/07.5K.jpg",
            "../img/08.6K.jpg"
    };

    // SERVER DINFO PATHS
    //const char *pathImg = "img/sheep.jpg";
    /*
    const char *outputPath = "img/results/";
    vector<string> pathImages = {
            "img/01.280x180.jpg",
            "img/02.299x160.jpg",
            "img/03.480p.jpg",
            "img/04.720p.jpg",
            "img/05.1080p.jpg",
            "img/06.4K.jpg",
            "img/07.5K.jpg",
            "img/08.6K.jpg"
    };
     */


    if (!filesystem::exists(outputPath))
        filesystem::create_directories(outputPath);

    //reqComp 0=auto, 1=gray scale, 3=RGB
    int reqComp = 0;
    vector<vector<unsigned char>> images;
    vector<int> widths;
    vector<int> heights;
    vector<int> channels;
    vector<string> names;
    for(const string& path : pathImages) {
        int width, height, currChannels;
        unsigned char *image = stbi_load(path.c_str(), &width, &height, &currChannels, 0);
        if (!image) {
            cerr << "Error: can't open image!" << endl;
            return -1;
        }
        images.emplace_back(image, image + width * height * currChannels);
        widths.push_back(width);
        heights.push_back(height);
        channels.push_back(currChannels);

        string name = getFileName(path);
        names.push_back(name);

        stbi_image_free(image);
    }

    cout << "Computing Sequential Kernel Image Processing ..." << endl;

    for (size_t i = 0; i < images.size(); ++i) {
        unsigned char* image = images[i].data();
        int width = widths[i];
        int height = heights[i];
        int currChannels = channels[i];
        string name = names[i];

        auto [outputImage, executionTime] = applyKernel(image, width, height, kernel, currChannels);
        string fullPath = outputPath + "seq_" + name + ".jpg";
        stbi_write_jpg(fullPath.c_str(), width, height, channels, outputImage.data(), 100);
        executionTimes[name].emplace_back(executionTime);
    }

    return 0;/*
    for (const auto &[name, kernel] : kernels) {
        auto [outputImage, executionTime] = applyKernel(image, width, height, kernel, channels);
        stbi_write_jpg((outputPath + name + ".jpg").c_str(), width, height, channels, outputImage.data(), 100);
        executionTimes[name].emplace_back(executionTime);
    }

    cout << "Computing OMP Kernel Image Processing ..." << endl;

    for (const auto &[name, kernel] : kernels) {
            auto [outputImage, executionTime] = applyKernelOMP(image, width, height, kernel, channels);
            stbi_write_jpg((outputPath + name + "_omp.jpg").c_str(), width, height, channels, outputImage.data(), 100);
            executionTimes[name].emplace_back(executionTime);
    }

    cout << "Computing CUDA Kernel Image Processing ..." << endl;

    for (const auto &[name, kernel] : kernels) {
        auto [outputImage, executionTime] = applyKernelCUDA(image, width, height, kernel, channels);
        stbi_write_jpg((outputPath + name + "_cuda.jpg").c_str(), width, height, channels, outputImage.data(), 100);
        executionTimes[name].emplace_back(executionTime);
    }


    printExecutionTimes(executionTimes);

    return 0;*/
}
