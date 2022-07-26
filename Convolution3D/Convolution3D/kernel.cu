#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <random>

#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define BLOCK_SIZE 8

void serialConvolve3D(int* input, int* output, int width, int height, int channel, int mask[3][3][3]);
void checkCorrect(int* groundTruth, int* test, int width, int height, int channel);
void printMatrix(int* data, int width, int height, int channel);
cudaError_t Convolution3D(int* input, int* output, int width, int height, int channel, int mask[3][3][3]);

__constant__ int MASK[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void convolution3DKernel(int* input, int* output, int width, int height, int channel) {
	__shared__ int tile[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

	int output_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int output_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int output_channel = blockIdx.z * BLOCK_SIZE + threadIdx.z;

	if (output_row >= 0 && output_row < height &&
		output_col >= 0 && output_col < width &&
		output_channel >= 0 && output_channel < channel) {
		tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[width * height * output_channel + width * output_row + output_col];
	} else {
		tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
	}
	__syncthreads();

	int this_tile_row_start = blockIdx.y * blockDim.y;
	int this_tile_col_start = blockIdx.x * blockDim.x;
	int this_tile_channel_start = blockIdx.z * blockDim.z;

	int next_tile_row_start = (blockIdx.y + 1) * blockDim.y;
	int next_tile_col_start = (blockIdx.x + 1) * blockDim.x;
	int next_tile_channel_start = (blockIdx.z + 1) * blockDim.z;

	int row_start_point = output_row - MASK_RADIUS;
	int col_start_point = output_col - MASK_RADIUS;
	int channel_start_point = output_channel - MASK_RADIUS;

	int sum = 0;
	for (int k_channel = 0; k_channel < MASK_WIDTH; k_channel++) {
		for (int k_row = 0; k_row < MASK_WIDTH; k_row++) {
			for (int k_col = 0; k_col < MASK_WIDTH; k_col++) {
				int row_idx = row_start_point + k_row;
				int col_idx = col_start_point + k_col;
				int channel_idx = channel_start_point + k_channel;

				if ((channel_idx >= 0 && channel_idx < channel) &&
					(row_idx >= 0 && row_idx < height) &&
					(col_idx >= 0 && col_idx < width)) {
					if (channel_idx >= this_tile_channel_start && channel_idx < next_tile_channel_start &&
						row_idx >= this_tile_row_start && row_idx < next_tile_row_start &&
						col_idx >= this_tile_col_start && col_idx < next_tile_col_start) {
						sum += MASK[k_channel][k_row][k_col] * tile[threadIdx.z - MASK_RADIUS + k_channel][threadIdx.y - MASK_RADIUS + k_row][threadIdx.x - MASK_RADIUS + k_col];
					} else {
						sum += MASK[k_channel][k_row][k_col] * input[(width * height) * channel_idx + width * row_idx + col_idx];
					}
				}
			}
		}
	}

	if ((output_row >= 0 && output_row < height) &&
		(output_col >= 0 && output_col < width) &&
		(output_channel >= 0 && output_channel < channel)) {
		output[(width * height) * output_channel + width * output_row + output_col] = sum;
	}
}

int main() {
	cudaError_t cudaStatus;
	int height = 22;
	int width = 18;
	int channel = 24;
	int mask[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH] = {
		{{0, 1, 0}, {1, 2, 1}, {0, 1, 0}},
		{{1, 2, 1}, {2, 3, 2}, {1, 2, 1}},
		{{0, 1, 0}, {1, 2, 1}, {0, 1, 0}}
	};

	int* input = new int[height * width * channel];
	int* outputSerial = new int[height * width * channel];
	int* outputParallel = new int[height * width * channel];

	// Randomly Generating input Array
	srand(time(NULL));
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				input[(width * height) * k + width * i + j] = rand() % 10;
			}
		}
	}
	printf("Finished Generating Input Array.\n");
	//printMatrix(input, width, height, channel);

	// Convolution 3D serial
	serialConvolve3D(input, outputSerial, width, height, channel, mask);
	printf("Finished Convolution 3D in serial.\n");
	//printMatrix(outputSerial, width, height, channel);

	// Convolution 3D in parallel
	cudaStatus = Convolution3D(input, outputParallel, width, height, channel, mask);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Convolution3D failed!");
		return 1;
	}
	printf("Finished Convolution 3D in parallel.\n");
	//printMatrix(outputParallel, width, height, channel);
	checkCorrect(outputSerial, outputParallel, width, height, channel);

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

// Convolution 3D in parallel.
cudaError_t Convolution3D(int* input, int* output, int width, int height, int channel, int mask[3][3][3]) {
	int* dev_in = 0;
	int* dev_out = 0;

	cudaError_t cudaStatus;

	// Allocate device memory for input and output
	cudaStatus = cudaMalloc((void**)&dev_in, width * height * channel * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, width * height * channel * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host to device.
	cudaStatus = cudaMemcpy(dev_in, input, width * height * channel * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy Convolution Mask from host to constant memory on device
	cudaStatus = cudaMemcpyToSymbol(MASK, mask, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}

	dim3 dimBlock;
	dim3 dimGrid;
	// Launch kernel
	dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dimGrid = dim3(ceil(1.0 * width / BLOCK_SIZE), ceil(1.0 * height / BLOCK_SIZE), ceil(1.0 * channel / BLOCK_SIZE));
	convolution3DKernel<<<dimGrid, dimBlock>>>(dev_in, dev_out, width, height, channel);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convolution3DKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching convolution3DKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from device to host memory.
	cudaStatus = cudaMemcpy(output, dev_out, width * height * channel * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);

	return cudaStatus;
}

void serialConvolve3D(int* input, int* output, int width, int height, int channel, int mask[3][3][3]) {
	for (int curr_channel = 0; curr_channel < channel; curr_channel++) {
		for (int curr_row = 0; curr_row < height; curr_row++) {
			for (int curr_col = 0; curr_col < width; curr_col++) {
				int currSum = 0;
				for (int k_channel = -MASK_RADIUS; k_channel <= MASK_RADIUS; k_channel++) {
					for (int k_row = -MASK_RADIUS; k_row <= MASK_RADIUS; k_row++) {
						for (int k_col = -MASK_RADIUS; k_col <= MASK_RADIUS; k_col++) {
							if (curr_row + k_row >= 0 && curr_row + k_row < height &&
								curr_col + k_col >= 0 && curr_col + k_col < width &&
								curr_channel + k_channel >= 0 && curr_channel + k_channel < channel) {
								currSum += input[(width * height) * (curr_channel + k_channel) + width * (curr_row + k_row) + curr_col + k_col] * mask[MASK_RADIUS + k_channel][MASK_RADIUS + k_row][MASK_RADIUS + k_col];
							}
						}
					}
				}
				output[(width * height) * curr_channel + curr_row * width + curr_col] = currSum;
			}
		}
	}
}

void checkCorrect(int* groundTruth, int* test, int width, int height, int channel) {
	bool correct = true;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (groundTruth[(width * height) * k + width * i + j] != test[(width * height) * k + width * i + j]) {
					correct = false;
				}
			}
		}
	}
	if (correct) printf("Correct Result!\n");
	else printf("Wrong Result!\n");
}

void printMatrix(int* data, int width, int height, int channel) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				printf("%d ", data[(width * height) * k + width * i + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}