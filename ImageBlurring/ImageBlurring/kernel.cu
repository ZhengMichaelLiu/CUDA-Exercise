
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <random>

#define BLUR_SIZE 1
#define BLOCK_WIDTH 32

void getSequentialResult(int* inputImg, int* outputImg, const int w, const int h);
cudaError_t imageBlur(int* in, int* out, const int w, const int h);
void printImage(int* img, int width, int height);

__global__ void imgBlurKernel(int* in, int* out, const int w, const int h) {
    // Use each thread to handle each output pixel
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only threads within valid range need to work
    if (col < w && row < h) {
        int accumulate = 0;
        int numPixels = 0;
        for (int blurRowIdx = -BLUR_SIZE; blurRowIdx <= BLUR_SIZE; blurRowIdx++) {
            for (int blurColIdx = -BLUR_SIZE; blurColIdx <= BLUR_SIZE; blurColIdx++) {
                int currRow = row + blurRowIdx;
                int currCol = col + blurColIdx;
                
                // Only calculate valid pixels within the image
                if (currRow >= 0 && currRow < h && currCol >= 0 && currCol < w) {
                    accumulate += in[currRow * w + currCol];
                    numPixels++;
                }
            }
        }
        
        // Write output
        out[row * w + col] = (int)(accumulate / numPixels);
    }
}

int main() {
    const int width = 1920;
    const int height = 1080;

    int size = width * height * sizeof(int);
    int* inputImg = (int*) malloc(size);
    int* outputImg = (int*) malloc(size);
    
    // Create input image randomly
    // The image may not be meaningful, just for test
    srand(time(NULL));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            inputImg[i * width + j] = rand() % 256;
        }
    }
    printf("Finished Generating Input.\n");
    // For debug only
    //printImage(inputImg, width, height);

    // Sequential result, for checking the parallel result later
    int* sequentialResult = (int*) malloc(size);
    getSequentialResult(inputImg, sequentialResult, width, height);
    printf("Finished Calculating Result using Sequential method.\n");
    // For debug only
    //printImage(sequentialResult, width, height);

    // Blurring Image in parallel.
    cudaError_t cudaStatus = imageBlur(inputImg, outputImg, width, height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Image Blur with CUDA failed!");
        return 1;
    }
    printf("Finished Calculating Result using Parallel method.\n");
    // For debug only
    //printImage(outputImg, width, height);

    // Check if the result is correct
    bool correct = true;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (sequentialResult[i * width + j] - outputImg[i * width + j]) {
                correct = false;
            }
        }
    }

    if (correct) {
        printf("Correct Result!\n");
    } else {
        printf("Wrong Result!\n");
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    free(inputImg);
    free(outputImg);
    free(sequentialResult);

    return 0;
}

// Use sequential method to get output.
// Compare to the parallel method to check if it is correct
void getSequentialResult(int* inputImg, int* outputImg, const int width, const int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int accumulate = 0;
            int numPixels = 0;
            for (int blurRowIdx = -BLUR_SIZE; blurRowIdx <= BLUR_SIZE; blurRowIdx++) {
                for (int blurColIdx = -BLUR_SIZE; blurColIdx <= BLUR_SIZE; blurColIdx++) {
                    int currRow = i + blurRowIdx;
                    int currCol = j + blurColIdx;
                    if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
                        accumulate += inputImg[currRow * width + currCol];
                        numPixels++;
                    }
                }
            }
            outputImg[i * width + j] = (int)(accumulate / numPixels);
        }
    }
}


// Helper function for using CUDA to blur image in parallel.
cudaError_t imageBlur(int* in, int *out, const int w, const int h) {
    cudaError_t cudaStatus;
    int size = w * h * sizeof(int);
    int* dev_in = 0;
    int* dev_out = 0;

    // Allocate GPU memory for input image
    cudaStatus = cudaMalloc((void**)&dev_in, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Allocate GPU memory for output image
    cudaStatus = cudaMalloc((void**)&dev_out, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input image from host to device.
    cudaStatus = cudaMemcpy(dev_in, in, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel.
    // 2D grid and 2D thread blocks
    dim3 dimGrid(ceil(1.0 * w / BLOCK_WIDTH), ceil(1.0 * h / BLOCK_WIDTH), 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    imgBlurKernel<<<dimGrid, dimBlock>>>(dev_in, dev_out, w, h);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "imgBlurKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching imgBlurKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_in);
    cudaFree(dev_out);
    
    return cudaStatus;
}

void printImage(int* img, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", img[i * width + j]);
        }
        printf("\n");
    }
}