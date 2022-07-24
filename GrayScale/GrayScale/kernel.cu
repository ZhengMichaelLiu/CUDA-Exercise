
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <random>

cudaError_t grayScaleWithCuda(uint8_t* inputImage, uint8_t* outputImage, int imageWidth, int imageHeight, int imageChannels);
void grayScaleSerial(uint8_t* inputImage, uint8_t* outputImage, int imageWidth, int imageHeight, int imageChannels);

__global__ void RGBtoGrayScale(uint8_t* ucharImg, uint8_t* grayImg, int width, int height) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < width && row < height) {
		int idx = width * row + col;
		uint8_t r = ucharImg[3 * idx];
		uint8_t g = ucharImg[3 * idx + 1];
		uint8_t b = ucharImg[3 * idx + 2];
		grayImg[idx] = (uint8_t)(0.21 * r + 0.71 * g + 0.07 * b);
	}
}

int main() {
	int imageWidth = 1920;
	int imageHeight = 1080;
	int imageChannels = 3;

	uint8_t* inputImage = new uint8_t[imageWidth * imageHeight * imageChannels];
	uint8_t* outputImage = new uint8_t[imageWidth * imageHeight];
	uint8_t* outputImageSerial = new uint8_t[imageWidth * imageHeight];

	// Randomly generate input image
	srand(time(NULL));
	for (int i = 0; i < imageWidth * imageHeight * imageChannels; i++) {
		inputImage[i] = (uint8_t)(rand() / RAND_MAX);
	}
	printf("Finished Generating Random Input Image.\n");

	// Serial
	grayScaleSerial(inputImage, outputImageSerial, imageWidth, imageHeight, imageChannels);
	printf("Finished Serial Image Grayscale.\n");

	// Parallel transfer image to grayscale
	cudaError_t cudaStatus = grayScaleWithCuda(inputImage, outputImage, imageWidth, imageHeight, imageChannels);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "grayScaleWithCuda failed!");
		return 1;
	}
	printf("Finished Parallel Image Grayscale.\n");

	// check if correct
	bool correct = true;
	for (int i = 0; i < imageWidth * imageHeight; i++) {
		if (abs(outputImageSerial[i] - outputImage[i]) > 0.001) {
			correct = false;
		}
	}
	if (correct) printf("Correct Result!\n");
	else printf("Wrong Result!\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	delete(inputImage);
	delete(outputImage);
	delete(outputImageSerial);

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t grayScaleWithCuda(uint8_t* inputImage, uint8_t* outputImage, int imageWidth, int imageHeight, int imageChannels) {
	uint8_t* dev_inImg = 0;
	uint8_t* dev_outImg = 0;
	cudaError_t cudaStatus;

	// Allocate GPU memory for input image
	cudaStatus = cudaMalloc((void**)&dev_inImg, imageWidth * imageHeight * imageChannels * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU memory for output image
	cudaStatus = cudaMalloc((void**)&dev_outImg, imageWidth * imageHeight * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input image from host to device.
	cudaStatus = cudaMemcpy(dev_inImg, inputImage, imageWidth * imageHeight * imageChannels * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(ceil(1.0 * imageWidth / 32.0), ceil(1.0 * imageHeight / 32.0), 1);
	RGBtoGrayScale<<<dimGrid, dimBlock>>>(dev_inImg, dev_outImg, imageWidth, imageHeight);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "RGBtoGrayScale launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching RGBtoGrayScale!\n", cudaStatus);
		goto Error;
	}

	// Copy output image from device to host.
	cudaStatus = cudaMemcpy(outputImage, dev_outImg, imageWidth * imageHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_inImg);
	cudaFree(dev_outImg);

	return cudaStatus;
}

void grayScaleSerial(uint8_t* inputImage, uint8_t* outputImage, int imageWidth, int imageHeight, int imageChannels) {
	for (int i = 0; i < imageHeight; i++) {
		for (int j = 0; j < imageWidth; j++) {
			int idx = i * imageWidth + j;
			uint8_t r = inputImage[3 * idx];
			uint8_t	g = inputImage[3 * idx + 1];
			uint8_t b = inputImage[3 * idx + 2];
			outputImage[idx] = (uint8_t)(0.21 * r + 0.71 * g + 0.07 * b);
		}
	}
}