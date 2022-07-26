#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <random>

#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define BLOCK_SIZE 16

void serialConvolve2D(int* input, int* output, int width, int height, int mask[3][3]);
void checkCorrect(int* groundTruth, int* test, int width, int height);
void printMatrix(int* data, int width, int height);

cudaError_t Convolution2D(int* input, int* output, int width, int height, int mask[3][3]);

__constant__ int MASK[MASK_WIDTH][MASK_WIDTH];

__global__ void convolution2DKernel(int* input, int* output, int width, int height) {
	__shared__ int tile[BLOCK_SIZE + MASK_WIDTH - 1][BLOCK_SIZE + MASK_WIDTH - 1];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row_o = blockIdx.y * BLOCK_SIZE + ty;
	int col_o = blockIdx.x * BLOCK_SIZE + tx;

	int row_i = row_o - MASK_RADIUS;
	int col_i = col_o - MASK_RADIUS;

	if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
		tile[ty][tx] = input[row_i * width + col_i];
	} else {
		tile[ty][tx] = 0;
	}
	__syncthreads();

	// some threads do not participate in calculating output 
	float sum = 0;
	if (ty < BLOCK_SIZE && tx < BLOCK_SIZE) {
		for (int i = 0; i < MASK_WIDTH; i++) {
			for (int j = 0; j < MASK_WIDTH; j++) {
				sum += MASK[i][j] * tile[i + ty][j + tx];
			}
		}

		// some threads do not write output 
		if (row_o < height && col_o < width) {
			output[row_o * width + col_o] = sum;
		}
	}
}

int main() {
	cudaError_t cudaStatus;
	int height = 17;
	int width = 17;
	int mask[MASK_WIDTH][MASK_WIDTH] = {
		{0, 1, 0},
		{1, 2, 1},
		{0, 1, 0}
	};

	int* input = new int[height * width];
	int* outputSerial = new int[height * width];
	int* outputParallel = new int[height * width];

	// Randomly Generating input Array
	srand(time(NULL));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			input[i * width + j] = rand() % 10;
		}
	}
	printf("Finished Generating Input Array.\n");
	printMatrix(input, width, height);

	// Convolution 2D serial
	serialConvolve2D(input, outputSerial, width, height, mask);
	printf("Finished Convolution 2D in serial.\n");
	printMatrix(outputSerial, width, height);

	// Convolution 2D in parallel
	cudaStatus = Convolution2D(input, outputParallel, width, height, mask);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Convolution1D Strategy1 failed!");
		return 1;
	}
	printf("Finished Convolution 2D in parallel.\n");
	printMatrix(outputParallel, width, height);
	checkCorrect(outputSerial, outputParallel, width, height);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	delete(input);
	delete(outputSerial);
	delete(outputParallel);

	return 0;
}

// Convolution 2D in parallel.
cudaError_t Convolution2D(int* input, int* output, int width, int height, int mask[3][3]) {
	int* dev_in = 0;
	int* dev_out = 0;

	cudaError_t cudaStatus;

	// Allocate device memory for input and output
	cudaStatus = cudaMalloc((void**)&dev_in, width * height * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, width * height * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host to device.
	cudaStatus = cudaMemcpy(dev_in, input, width * height * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy Convolution Mask from host to constant memory on device
	cudaStatus = cudaMemcpyToSymbol(MASK, mask, MASK_WIDTH * MASK_WIDTH * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}

	dim3 dimBlock;
	dim3 dimGrid;
	// Launch kernel
	dimBlock = dim3(BLOCK_SIZE + MASK_WIDTH - 1, BLOCK_SIZE + MASK_WIDTH - 1, 1);
	dimGrid = dim3(ceil(1.0 * width / BLOCK_SIZE), ceil(1.0 * height / BLOCK_SIZE), 1);
	convolution2DKernel<<<dimGrid, dimBlock>>>(dev_in, dev_out, width, height);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convolution2DKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching convolution2DKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from device to host memory.
	cudaStatus = cudaMemcpy(output, dev_out, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);

	return cudaStatus;
}

void serialConvolve2D(int* input, int* output, int width, int height, int mask[3][3]) {
	for (int curr_row = 0; curr_row < height; curr_row++) {
		for (int curr_col = 0; curr_col < width; curr_col++) {
			int currSum = 0;
			for (int k_row = -MASK_RADIUS; k_row <= MASK_RADIUS; k_row++) {
				for (int k_col = -MASK_RADIUS; k_col <= MASK_RADIUS; k_col++) {
					if (curr_row + k_row >= 0 && curr_row + k_row < height && curr_col + k_col >= 0 && curr_col + k_col < width) {
						currSum += input[(curr_row + k_row) * width + curr_col + k_col] * mask[(MASK_RADIUS + k_row)][(MASK_RADIUS + k_col)];
					}
				}
			}
			output[curr_row * width + curr_col] = currSum;
		}
	}
}

void checkCorrect(int* groundTruth, int* test, int width, int height) {
	bool correct = true;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (groundTruth[i * width + j] != test[i * width + j]) {
				correct = false;
			}
		}
	}
	if (correct) printf("Correct Result!\n");
	else printf("Wrong Result!\n");
}

void printMatrix(int* data, int width, int height) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%d ", data[i * width + j]);
		}
		printf("\n");
	}
	printf("\n");
}