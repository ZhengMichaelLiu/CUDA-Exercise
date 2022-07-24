
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <random>
#include <algorithm>

#define HISTO_LENGTH 256
#define BLOCK_SIZE 32

cudaError_t histogramWithCuda(float* inputImage, float* outputImage, int imageWidth, int imageHeight, int imageChannels);
void histogramSerial(float* inputImage, float* outputImage, int imageWidth, int imageHeight, int imageChannels);

__global__ void castImageFloatToUnsignedChar(float* inImg, uint8_t* outImg, int width, int height) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < width && row < height) {
		int ii = (width * height) * blockIdx.z + width * row + col;
		outImg[ii] = (uint8_t)(255 * inImg[ii]);
	}
}

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

__global__ void histogramOfGrayImg(uint8_t* grayImg, int width, int height, int* histogram) {
	// declare space for private histogram for each block in shared mem 
	__shared__ int histo_private[HISTO_LENGTH];
	int tIdx = blockDim.x * threadIdx.y + threadIdx.x;
	if (tIdx < HISTO_LENGTH) {
		histo_private[tIdx] = 0;
	}
	__syncthreads();

	// build the private histogram, same as before, but using shared mem   
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < width && row < height) {
		int idx = row * width + col;
		atomicAdd(&(histo_private[grayImg[idx]]), 1);
	}

	// wait all threads in the block to finish   
	__syncthreads();

	// build the global histrogram   
	if (tIdx < HISTO_LENGTH) {
		atomicAdd(&(histogram[tIdx]), histo_private[tIdx]);
	}
}

__global__ void CDFofHistogram(int* histogram, float* cdf, int width, int height) {

	// load data into shared memory, each block loads BLOCK_SIZE * 2 elements
	__shared__ int T[HISTO_LENGTH];

	int i = threadIdx.x;

	if (i < HISTO_LENGTH) {
		T[i] = histogram[i];
	}

	// Scan step
	for (int stride = 1; stride <= blockDim.x / 2; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < blockDim.x && (index - stride) >= 0) {
			T[index] += T[index - stride];
		}
	}

	// Post Scan Step
	for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < HISTO_LENGTH) {
			T[index + stride] += T[index];
		}
	}

	// Write back to output
	__syncthreads();
	if (i < HISTO_LENGTH) {
		cdf[i] = (1.0 * T[i]) / ((float)(width * height));
	}
}

__global__ void histogramEqualization(uint8_t* ucharImg, float* cdf, int width, int height) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < width && row < height) {
		int i = width * height * blockIdx.z + width * row + col;
		float v = 255 * (cdf[ucharImg[i]] - cdf[0]) / (1.0 - cdf[0]);
		v = min(max(v, 0.0f), 255.0f);
		ucharImg[i] = (uint8_t)v;
	}
}

__global__ void castImageUnsignedCharToFloat(uint8_t* inImg, float* outImg, int width, int height) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < width && row < height) {
		int i = (width * height) * blockIdx.z + width * row + col;
		outImg[i] = (float)(inImg[i] / 255.0);
	}
}

int main() {
	int imageWidth = 10;
	int imageHeight = 10;
	int imageChannels = 3;

	float* inputImage = new float[imageWidth * imageHeight * imageChannels];
	float* outputImage = new float[imageWidth * imageHeight * imageChannels];
	float* serialOutputImage = new float[imageWidth * imageHeight * imageChannels];

	// Randomly generate input image
	srand(time(NULL));
	for (int i = 0; i < imageWidth * imageHeight * imageChannels; i++) {
		inputImage[i] = (float)(1.0 * rand() / RAND_MAX);
	}
	printf("Finished generating random input image. \n");
	//for (int i = 0; i < imageWidth * imageHeight * imageChannels; i++) {
	//	printf("%f ", inputImage[i]);
	//}

	// serial
	histogramSerial(inputImage, serialOutputImage, imageWidth, imageHeight, imageChannels);
	printf("Finished generating serial output image. \n");
	/*for (int i = 0; i < imageWidth * imageHeight * imageChannels; i++) {
		printf("%f ", serialOutputImage[i]);
	}*/

	// parallel
	cudaError_t cudaStatus = histogramWithCuda(inputImage, outputImage, imageWidth, imageHeight, imageChannels);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "histogramWithCuda failed!");
		return 1;
	}
	//for (int i = 0; i < imageWidth * imageHeight * imageChannels; i++) {
	//	printf("%f ", outputImage[i]);
	//}

	// check if correct
	bool correct = true;
	for (int i = 0; i < imageWidth * imageHeight * imageChannels; i++) {
		if (abs(outputImage[i] - serialOutputImage[i]) > 0.01) {
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
	delete(serialOutputImage);
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t histogramWithCuda(float* inputImage, float* outputImage, int imageWidth, int imageHeight, int imageChannels) {
	float* dev_inImg = 0;
	uint8_t* dev_inImgUChar = 0;
	uint8_t* dev_imgGray = 0;
	int* dev_histogram = 0;
	float* dev_cdf = 0;
	float* dev_outImg = 0;

	cudaError_t cudaStatus;

	// Allocate GPU memory for input image
	cudaStatus = cudaMalloc((void**)&dev_inImg, imageWidth * imageHeight * imageChannels * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU memory for input image uint8_t
	cudaStatus = cudaMalloc((void**)&dev_inImgUChar, imageWidth * imageHeight * imageChannels * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU memory for gray image
	cudaStatus = cudaMalloc((void**)&dev_imgGray, imageWidth * imageHeight * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU memory for histogram
	cudaStatus = cudaMalloc((void**)&dev_histogram, HISTO_LENGTH * sizeof(int));
	cudaStatus = cudaMemset((void*)dev_histogram, 0, HISTO_LENGTH * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU memory for cdf
	cudaStatus = cudaMalloc((void**)&dev_cdf, HISTO_LENGTH * sizeof(float));
	cudaStatus = cudaMemset((void*)dev_cdf, 0, HISTO_LENGTH * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU memory for output image
	cudaStatus = cudaMalloc((void**)&dev_outImg, imageWidth * imageHeight * imageChannels * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input image from host to device.
	cudaStatus = cudaMemcpy(dev_inImg, inputImage, imageWidth * imageHeight * imageChannels * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 dimGrid;
	dim3 dimBlock;
	// ===========================================================================================
	// Cast the image from float to uint8_t
	dimGrid = dim3(ceil(1.0 * imageWidth / BLOCK_SIZE), ceil(1.0 * imageHeight / BLOCK_SIZE), imageChannels);
	dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	castImageFloatToUnsignedChar<<<dimGrid, dimBlock>>>(dev_inImg, dev_inImgUChar, imageWidth, imageHeight);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "castImageFloatToUnsignedChar launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching castImageFloatToUnsignedChar!\n", cudaStatus);
		goto Error;
	}

	// ===========================================================================================
	// Convert the image from RGB to GrayScale
	dimGrid = dim3(ceil(1.0 * imageWidth / BLOCK_SIZE), ceil(1.0 * imageHeight / BLOCK_SIZE), 1);
	dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	RGBtoGrayScale<<<dimGrid, dimBlock>>>(dev_inImgUChar, dev_imgGray, imageWidth, imageHeight);

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

	// ===========================================================================================
	// Compute the histogram of grayImage
	dimGrid = dim3(ceil(1.0 * imageWidth / BLOCK_SIZE), ceil(1.0 * imageHeight / BLOCK_SIZE), 1);
	dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	histogramOfGrayImg<<<dimGrid, dimBlock>>>(dev_imgGray, imageWidth, imageHeight, dev_histogram);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "histogramOfGrayImg launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching histogramOfGrayImg!\n", cudaStatus);
		goto Error;
	}

	// ===========================================================================================
	// Compute the Cumulative Distribution Function of histogram
	dimGrid = dim3(1, 1, 1);
	dimBlock = dim3(HISTO_LENGTH, 1, 1);
	CDFofHistogram<<<dimGrid, dimBlock>>>(dev_histogram, dev_cdf, imageWidth, imageHeight);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CDFofHistogram launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CDFofHistogram!\n", cudaStatus);
		goto Error;
	}

	// ===========================================================================================
	// Histogram equalization
	dimGrid = dim3(ceil(1.0 * imageWidth / BLOCK_SIZE), ceil(1.0 * imageHeight / BLOCK_SIZE), imageChannels);
	dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	histogramEqualization<<<dimGrid, dimBlock>>>(dev_inImgUChar, dev_cdf, imageWidth, imageHeight);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "histogramEqualization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching histogramEqualization!\n", cudaStatus);
		goto Error;
	}

	// ===========================================================================================
	// Cast back to float
	dimGrid = dim3(ceil(1.0 * imageWidth / BLOCK_SIZE), ceil(1.0 * imageHeight / BLOCK_SIZE), imageChannels);
	dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	castImageUnsignedCharToFloat<<<dimGrid, dimBlock>>>(dev_inImgUChar, dev_outImg, imageWidth, imageHeight);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "castImageUnsignedCharToFloat launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching castImageUnsignedCharToFloat!\n", cudaStatus);
		goto Error;
	}


	// Copy output image from device to host.
	cudaStatus = cudaMemcpy(outputImage, dev_outImg, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_inImg);
	cudaFree(dev_inImgUChar);
	cudaFree(dev_imgGray);
	cudaFree(dev_histogram);
	cudaFree(dev_cdf);
	cudaFree(dev_outImg);

	return cudaStatus;
}

void histogramSerial(float* inputImage, float* outputImage, int imageWidth, int imageHeight, int imageChannels) {
	// Cast the image from float to uint8_t
	uint8_t* ucharImage = new uint8_t[imageWidth * imageHeight * imageChannels];
	for (int ii = 0; ii < (imageWidth * imageHeight * imageChannels); ii++) {
		ucharImage[ii] = (uint8_t)(255 * inputImage[ii]);
	}

	// Convert the image from RGB to GrayScale
	uint8_t* grayImage = new uint8_t[imageWidth * imageHeight];
	for (int ii = 0; ii < imageHeight; ii++) {
		for (int jj = 0; jj < imageWidth; jj++) {
			int idx = ii * imageWidth + jj;
			uint8_t r = ucharImage[3 * idx];
			uint8_t g = ucharImage[3 * idx + 1];
			uint8_t	b = ucharImage[3 * idx + 2];
			grayImage[idx] = (uint8_t)(0.21 * r + 0.71 * g + 0.07 * b);
		}
	}

	// Compute the histogram of grayImage
	int* histogram = new int[256];
	for (int i = 0; i < 256; i++) {
		histogram[i] = 0;
	}
	for (int i = 0; i < imageWidth * imageHeight; i++) {
		histogram[grayImage[i]]++;
	}

	// Compute the Cumulative Distribution Function of histogram
	float* cdf = new float[256];
	cdf[0] = (1.0 * histogram[0]) / (imageWidth * imageHeight);
	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i - 1] + (1.0 * histogram[i]) / (imageWidth * imageHeight);
	}

	// histogram equalization
	for (int i = 0; i < imageWidth * imageHeight * imageChannels; i++) {
		ucharImage[i] = (uint8_t)std::min(std::max((255 * (cdf[ucharImage[i]] - cdf[0]) / (1.0 - cdf[0])), 0.0), 255.0);
	}

	// Cast back to float
	for (int i = 0; i < imageWidth * imageChannels * imageHeight; i++) {
		outputImage[i] = (float)(ucharImage[i] / 255.0);
	}
}