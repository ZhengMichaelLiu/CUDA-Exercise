
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <random>

#define BLOCK_SIZE 512

cudaError_t listReductionWithCuda(int* inputArray, int* outputArray, int inputSize, int outputSize);

__global__ void listReductionKernel(int* inputArray, int* outputArray, int inputSize) {

	// use one block to process 2 * BLOCK_SIZE elements
	__shared__ int partialSum[2 * BLOCK_SIZE];

	int i = threadIdx.x;
	int start = 2 * blockDim.x * blockIdx.x;

	// load 2 * BLOCK_SIZE elements in
	if (start + i < inputSize) {
		partialSum[i] = inputArray[start + i];
	} else {
		partialSum[i] = 0;
	}

	if (start + blockDim.x + i < inputSize) {
		partialSum[blockDim.x + i] = inputArray[start + blockDim.x + i];
	} else {
		partialSum[blockDim.x + i] = 0;
	}

	// parallel reduction
	for (int stride = blockDim.x; stride >= 1; stride /= 2) {
		__syncthreads();
		if (i < stride) {
			partialSum[i] += partialSum[i + stride];
		}
	}
	// write result to output
	if (i == 0) {
		outputArray[blockIdx.x] = partialSum[0];
	}
}

int main() {

	int inputArraySize = 45635231;
	int* inputArray = new int[inputArraySize];

	int outputArraySize = inputArraySize / (BLOCK_SIZE * 2);
	if (inputArraySize % (BLOCK_SIZE * 2)) {
		outputArraySize++;
	}
	int* outputArray = new int[outputArraySize];

	// Randomly Generate input array
	srand(time(NULL));
	for (int i = 0; i < inputArraySize; i++) {
		inputArray[i] = rand() % 3;
	}
	printf("Finished Generating Input Array.\n");

	// List Reduction in serial
	clock_t start = clock();
	int serialResult = 0;
	for (int i = 0; i < inputArraySize; i++) {
		serialResult += inputArray[i];
	}
	clock_t end = clock();
	double elapsed = double(end - start) / CLOCKS_PER_SEC;
	printf("Finished Calculating List Reduction in Serial: %d.\nTime Elapsed: %f seconds.\n", serialResult, elapsed);

	// List Reduction in parallel.
	cudaError_t cudaStatus = listReductionWithCuda(inputArray, outputArray, inputArraySize, outputArraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "listReductionWithCuda failed!");
		return 1;
	}

	int parallelResult = 0;
	for (int i = 0; i < outputArraySize; i++) {
		parallelResult += outputArray[i];
	}
	printf("Finished Calculating List Reduction in Parallel. %d\n", parallelResult);

	if (abs(parallelResult - serialResult) > 0.001) {
		printf("Wrong Result!\n");
	} else {
		printf("Correct Result!\n");
	}
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	delete(inputArray);
	delete(outputArray);

	return 0;
}

// List Reduction in parallel.
cudaError_t listReductionWithCuda(int* inputArray, int* outputArray, int inputSize, int outputSize) {
	int* dev_in = 0;
	int* dev_out = 0;
	cudaError_t cudaStatus;

	// Allocate device memory for input list
	cudaStatus = cudaMalloc((void**)&dev_in, inputSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate device memory for output list
	cudaStatus = cudaMalloc((void**)&dev_out, outputSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input list from host to device.
	cudaStatus = cudaMemcpy(dev_in, inputArray, inputSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel
	dim3 dimGrid(outputSize, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	listReductionKernel<<<dimGrid, dimBlock>>>(dev_in, dev_out, inputSize);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "listReductionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching listReductionKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from device to host.
	cudaStatus = cudaMemcpy(outputArray, dev_out, outputSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);

	return cudaStatus;
}