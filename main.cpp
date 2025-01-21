//
// Created by Francesco on 04/11/2024.
//
#include <iostream>
#include <vector>
#include <array>
#include <map>
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"
#include <chrono>
#include <omp.h>

using namespace std;

const array<array<float, 3>, 3> identity = {{
                                                    {0, 0, 0},
                                                    {0, 1, 0},
                                                    {0, 0, 0}
                                            }};

const array<array<float, 3>, 3> edge_detection = {{
                                                          {0, -1, 0},
                                                          {-1, 4, -1},
                                                          {0, -1, 0}
                                                  }};

const array<array<float, 3>, 3> edge_detection2 = {{
                                                          {-1, -1, -1},
                                                          {-1, 8, -1},
                                                          {-1, -1, -1}
                                                  }};

const array<array<float, 3>, 3> sharpen = {{
                                                   {0, -1, 0},
                                                   {-1, 5, -1},
                                                   {0, -1, 0}
                                           }};

const array<array<float, 3>, 3> box_blur = {{
                                                    {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
                                                    {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
                                                    {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}
                                            }};

const array<array<float, 3>, 3> gaussian_blur = {{
                                                    {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
                                                    {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
                                                    {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
                                            }};

const array<array<float, 5>, 5> gaussian_blur_5x5 = {{
                                                         {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f},
                                                         {4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
                                                         {6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f},
                                                         {4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f},
                                                         {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f}
                                                 }};

const array<array<float, 5>, 5> unsharp_masking_5x5 = {{
                                                         {-1 / 256.0f, -4 / 256.0f, -6 / 256.0f, - 4 / 256.0f, -1 / 256.0f},
                                                         {-4 / 256.0f, -16 / 256.0f, -24 / 256.0f, - 16 / 256.0f, -4 / 256.0f},
                                                         {-6 / 256.0f, -24 / 256.0f, 476 / 256.0f, -24 / 256.0f, -6 / 256.0f},
                                                         {-4 / 256.0f, -16 / 256.0f, -24 / 256.0f, -16 / 256.0f, -4 / 256.0f},
                                                         {-1 / 256.0f, -4 / 256.0f, -6 / 256.0f, -4 / 256.0f, -1 / 256.0f}
                                                 }};

// Mappa per associare i nomi ai kernel
map<string, array<array<float, 3>, 3>> kernels = {
        {"identity", identity},
        {"edge_detection", edge_detection},
        {"edge_detection2", edge_detection2},
        {"sharpen", sharpen},
        {"box_blur", box_blur},
        {"gaussian_blur", gaussian_blur}
};
map<string, array<array<float, 5>, 5>> kernels5x5 = {
        {"gaussian_blur_5x5", gaussian_blur_5x5},
        {"unsharp_masking_5x5", unsharp_masking_5x5}
};


vector<unsigned char> apply_kernel(
        const unsigned char *image, int width, int height,
        const array<array<float, 3>, 3> &kernel, int channels) {

    vector<unsigned char> outputImage(width * height * channels);
    int halfSize = 1;

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

    return outputImage;
}
vector<unsigned char> apply_kernel_5x5(
        const unsigned char *image, int width, int height,
        const array<array<float, 5>, 5> &kernel, int channels) {

    vector<unsigned char> outputImage(width * height * channels);
    int halfSize = 2;

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

    return outputImage;
}

vector<unsigned char> apply_kernel_parallel(
        const unsigned char *image, int width, int height,
        const array<array<float, 3>, 3> &kernel, int channels) {

    vector<unsigned char> outputImage(width * height * channels);
    int halfSize = 1;

    #pragma omp parallel for collapse(2)
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

    return outputImage;
}
vector<unsigned char> apply_kernel_5x5_parallel(
        const unsigned char *image, int width, int height,
        const array<array<float, 5>, 5> &kernel, int channels) {

    vector<unsigned char> outputImage(width * height * channels);
    int halfSize = 2;

    #pragma omp parallel for collapse(2)
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

    return outputImage;
}

// Dichiarazione della funzione CUDA (kernel) definita in kernel.cu
void run_kernel(const char* inputPath, const char* outputPath);

int main() {
    int width, height, channels;

    //const char *path_img = "../img/test.png";
    //const char *path_img = "../img/panda.jpg";
    const char *path_img = "../img/sheep.jpg";
    const char *output_path = "../img/results/";

    //req_comp 0=auto, 1=scala grigi, 3=RGB
    int req_comp = 0;

    unsigned char *image = stbi_load(path_img, &width, &height, &channels, req_comp);
    if (!image) {
        cerr << "Errore: Impossibile aprire l'immagine!" << endl;
        return -1;
    }

    cout << "Sequential Kernel Image Processing Started" << endl;
    auto start_time = chrono::high_resolution_clock::now();

    for (const auto& [name, kernel] : kernels) {
        auto start_time_tmp = chrono::high_resolution_clock::now();
        vector<unsigned char> outputImage = apply_kernel(image, width, height, kernel, channels);
        stbi_write_jpg((output_path + name + ".jpg").c_str(), width, height, channels, outputImage.data(), 100);
        auto end_time_tmp = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_seconds_tmp = end_time_tmp - start_time_tmp;
        cout << "Tempo di esecuzione per " << name << ": " << elapsed_seconds_tmp.count() << " s" << endl;
    }
    for (const auto& [name, kernel] : kernels5x5) {
        auto start_time_tmp = chrono::high_resolution_clock::now();
        vector<unsigned char> outputImage = apply_kernel_5x5(image, width, height, kernel, channels);
        stbi_write_jpg((output_path + name + ".jpg").c_str(), width, height, channels, outputImage.data(), 100);
        auto end_time_tmp = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_seconds_tmp = end_time_tmp - start_time_tmp;
        cout << "Tempo di esecuzione per " << name << ": " << elapsed_seconds_tmp.count() << " s" << endl;
    }

    auto end_time = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed_seconds = end_time - start_time;
    cout << "Tempo di esecuzione totale: " << elapsed_seconds.count() << " s" << endl;

    cout << "\nParallel Kernel Image Processing Started" << endl;
    start_time = chrono::high_resolution_clock::now();

    for (const auto& [name, kernel] : kernels) {
        auto start_time_tmp = chrono::high_resolution_clock::now();
        vector<unsigned char> outputImage = apply_kernel_parallel(image, width, height, kernel, channels);
        stbi_write_jpg((output_path + name + "_parallel.jpg").c_str(), width, height, channels, outputImage.data(), 100);
        auto end_time_tmp = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_seconds_tmp = end_time_tmp - start_time_tmp;
        cout << "Tempo di esecuzione per " << name << ": " << elapsed_seconds_tmp.count() << " s" << endl;
    }
    for (const auto& [name, kernel] : kernels5x5) {
        auto start_time_tmp = chrono::high_resolution_clock::now();
        vector<unsigned char> outputImage = apply_kernel_5x5_parallel(image, width, height, kernel, channels);
        stbi_write_jpg((output_path + name + "_parallel.jpg").c_str(), width, height, channels, outputImage.data(), 100);
        auto end_time_tmp = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_seconds_tmp = end_time_tmp - start_time_tmp;
        cout << "Tempo di esecuzione per " << name << ": " << elapsed_seconds_tmp.count() << " s" << endl;
    }

    end_time = chrono::high_resolution_clock::now();

    elapsed_seconds = end_time - start_time;
    cout << "Tempo di esecuzione totale: " << elapsed_seconds.count() << " s" << endl;


    stbi_image_free(image);

    run_kernel(path_img, output_path);

    return 0;
}
