#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <random>

#define MASK_WIDTH 5
#define MASK_RADIUS 2
#define BLOCK_SIZE 512

void serialConvolve1D(int* input, int* output, int numOfElement, int* mask);
void checkCorrect(int* groundTruth, int* test, int numOfElement);
void printArray(int* data, int n);

cudaError_t Convolution1D(int* input, int* output, int numOfElement, int* mask, int strategy);

__constant__ int MASK[MASK_WIDTH];

__global__ void convolution1DKernel_Strategy1(int* input, int* output, int numOfElement) {
	__shared__ int tile[MASK_WIDTH + BLOCK_SIZE - 1];
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	// Load left halo elements
	int halo_left_idx = blockDim.x * (blockIdx.x - 1) + threadIdx.x;
	if (threadIdx.x >= blockDim.x - MASK_RADIUS) {
		tile[threadIdx.x - (blockDim.x - MASK_RADIUS)] = (halo_left_idx < 0) ? 0.0 : input[halo_left_idx];
	}

	// Load center elements
	if (i < numOfElement) {
		tile[MASK_RADIUS + threadIdx.x] = input[i];
	}

	// Load right halo elements
	int halo_right_idx = blockDim.x * (blockIdx.x + 1) + threadIdx.x;
	if (threadIdx.x < MASK_RADIUS) {
		tile[MASK_RADIUS + blockDim.x + threadIdx.x] = (halo_right_idx > numOfElement) ? 0.0 : input[halo_right_idx];
	}

	__syncthreads();

	// Convolution
	int sum = 0;
	for (int k = 0; k < MASK_WIDTH; k++) {
		sum += tile[threadIdx.x + k] * MASK[k];
	}

	if (i < numOfElement) {
		output[i] = sum;
	}
}

__global__ void convolution1DKernel_Strategy2(int* input, int* output, int numOfElement) {
	__shared__ int tile[BLOCK_SIZE + MASK_WIDTH - 1];

	int tx = threadIdx.x;
	int output_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int input_idx = output_idx - MASK_RADIUS;

	if ((input_idx >= 0) && (input_idx < numOfElement)) {
		tile[tx] = input[input_idx];
	} else { 
		tile[tx] = 0;
	}
	__syncthreads();

	int sum = 0;
	if (tx < BLOCK_SIZE) {
		for (int i = 0; i < MASK_WIDTH; i++) {
			sum += MASK[i] * tile[i + tx];
		}

		if (output_idx < numOfElement) {
			output[output_idx] = sum;
		}
	}
}

__global__ void convolution1DKernel_Strategy3(int* input, int* output, int numOfElement) {
	__shared__ int tile[BLOCK_SIZE];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numOfElement) {
		tile[threadIdx.x] = input[i];
	}
	__syncthreads();

	int tileStartPoint = blockIdx.x * blockDim.x;
	int tileEndPoint = (blockIdx.x + 1) * blockDim.x - 1;
	int startPoint = i - MASK_RADIUS;

	int sum = 0;
	for (int k = 0; k < MASK_WIDTH; k++) {
		int currIdx = startPoint + k;
		if (currIdx >= 0 && currIdx < numOfElement) {
			if (currIdx >= tileStartPoint && currIdx < tileEndPoint) {
				sum += tile[threadIdx.x - MASK_RADIUS + k] * MASK[k];
			} else {
				sum += input[currIdx] * MASK[k];
			}
		}
	}
	if (i < numOfElement) {
		output[i] = sum;
	}
}

int main() {
	cudaError_t cudaStatus;
	const int numOfElement = 10000;
	int mask[MASK_WIDTH] = { 1, 2, 3, 2, 1 };
	int* maskptr = mask;
	int* input = new int[numOfElement];
	int* outputSerial = new int[numOfElement];
	int* outputStrategy1 = new int[numOfElement];
	int* outputStrategy2 = new int[numOfElement];
	int* outputStrategy3 = new int[numOfElement];

	// Randomly Generating input Array
	srand(time(NULL));
	for (int i = 0; i < numOfElement; i++) {
		input[i] = rand() % 10;
	}
	printf("Finished Generating Input Array.\n");
	//printArray(input, numOfElement);

	// Convolution 1D serial
	serialConvolve1D(input, outputSerial, numOfElement, maskptr);
	printf("Finished Convolution 1D in serial.\n");
	//printArray(outputSerial, numOfElement);

	// Convolution 1D in parallel. Strategy 1
	cudaStatus = Convolution1D(input, outputStrategy1, numOfElement, maskptr, 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Convolution1D Strategy1 failed!");
		return 1;
	}
	// check if strategy 1 is correct
	checkCorrect(outputSerial, outputStrategy1, numOfElement);
	//printArray(outputStrategy1, numOfElement);

	// Convolution 1D in parallel. Strategy 2
	cudaStatus = Convolution1D(input, outputStrategy2, numOfElement, maskptr, 2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Convolution1D Strategy2 failed!");
		return 1;
	}
	// check if strategy 2 is correct
	checkCorrect(outputSerial, outputStrategy2, numOfElement);
	//printArray(outputStrategy2, numOfElement);

	// Convolution 1D in parallel. Strategy 3
	cudaStatus = Convolution1D(input, outputStrategy3, numOfElement, maskptr, 3);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Convolution1D Strategy3 failed!");
		return 1;
	}
	// check if strategy 3 is correct
	checkCorrect(outputSerial, outputStrategy3, numOfElement);
	//printArray(outputStrategy3, numOfElement);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	delete(input);
	delete(outputSerial);
	delete(outputStrategy1);
	delete(outputStrategy2);
	delete(outputStrategy3);

	return 0;
}

// Convolution 1D in parallel.
cudaError_t Convolution1D(int* input, int* output, int numOfElement, int* mask, int strategy) {
	int* dev_in = 0;
	int* dev_out = 0;

	cudaError_t cudaStatus;

	// Allocate device memory for input and output
	cudaStatus = cudaMalloc((void**)&dev_in, numOfElement * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, numOfElement * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host to device.
	cudaStatus = cudaMemcpy(dev_in, input, numOfElement * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy Convolution Mask from host to constant memory on device
	cudaStatus = cudaMemcpyToSymbol(MASK, mask, MASK_WIDTH * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed!");
		goto Error;
	}

	dim3 dimBlock;
	dim3 dimGrid;
	// Launch kernels for different strategies.
	if (strategy == 1) {
		dimBlock = dim3(BLOCK_SIZE, 1, 1);
		dimGrid = dim3(ceil(1.0 * numOfElement / BLOCK_SIZE), 1, 1);
		convolution1DKernel_Strategy1<<<dimGrid, dimBlock>>>(dev_in, dev_out, numOfElement);
	}

	if (strategy == 2) {
		dimBlock = dim3(BLOCK_SIZE + MASK_WIDTH - 1, 1, 1);
		dimGrid = dim3(ceil(1.0 * numOfElement / BLOCK_SIZE), 1, 1);
		convolution1DKernel_Strategy2<<<dimGrid, dimBlock>>>(dev_in, dev_out, numOfElement);
	}

	if (strategy == 3) {
		dimBlock = dim3(BLOCK_SIZE, 1, 1);
		dimGrid = dim3(ceil(1.0 * numOfElement / BLOCK_SIZE), 1, 1);
		convolution1DKernel_Strategy3<<<dimGrid, dimBlock>>>(dev_in, dev_out, numOfElement);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convolution1DKernel Strategy %d launch failed: %s\n", strategy, cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching convolution1DKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from device to host memory.
	cudaStatus = cudaMemcpy(output, dev_out, numOfElement * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);

	return cudaStatus;
}

void serialConvolve1D(int* input, int* output, int numOfElement, int* mask) {
	for (int i = 0; i < numOfElement; i++) {
		int currSum = 0;
		for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
			if (i + j >= 0 && i + j < numOfElement) {
				currSum += input[i + j] * mask[j + MASK_RADIUS];
			}
		}
		output[i] = currSum;
	}
}

void checkCorrect(int* groundTruth, int* test, int numOfElement) {
	bool correct = true;
	for (int i = 0; i < numOfElement; i++) {
		if (groundTruth[i] != test[i]) {
			correct = false;
			break;
		}
	}
	if (correct) printf("Correct Result!\n");
	else printf("Wrong Result!\n");
}

void printArray(int* data, int n) {
	for (int i = 0; i < n; i++) {
		printf("%d ", data[i]);
	}
	printf("\n");
}