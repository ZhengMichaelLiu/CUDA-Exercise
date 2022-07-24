#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <random>

#define BLOCK_SIZE 1024

cudaError_t listScanWithCuda(int* inputArray, int* outputArray, int numOfElement);

__global__ void listScanKernel_BrentKung(int* inputArray, int* outputArray, int numOfElement, int* intermediateArray) {

	// load data into shared memory, each block loads BLOCK_SIZE * 2 elements
	__shared__ int T[2 * BLOCK_SIZE];

	int i = threadIdx.x;
	int start = 2 * blockDim.x * blockIdx.x;

	if (start + i < numOfElement) {
		T[i] = inputArray[start + i];
	} else {
		T[i] = 0;
	}

	if (start + i + blockDim.x < numOfElement) {
		T[i + blockDim.x] = inputArray[start + i + blockDim.x];
	} else {
		T[i + blockDim.x] = 0;
	}

	// Scan step
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
			T[index] += T[index - stride];
		}
	}

	// Post Scan Step

	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE) {
			T[index + stride] += T[index];
		}
	}

	// Write back to output
	__syncthreads();
	if (start + i < numOfElement) {
		outputArray[start + i] = T[i];
	}
	if (start + i + blockDim.x < numOfElement) {
		outputArray[start + i + blockDim.x] = T[i + blockDim.x];
	}

	// store partial sum
	if (threadIdx.x == 0) {
		intermediateArray[blockIdx.x] = T[2 * BLOCK_SIZE - 1];
	}

}

__global__ void addBlockPartialSumScanToEachBlockKernel(int* inputArray, int* eachBlocktoAdd, int numOfElement) {
	__shared__ int toAdd;

	if (threadIdx.x == 0) {
		if (blockIdx.x == 0) {
			toAdd = 0;
		} else {
			toAdd = eachBlocktoAdd[blockIdx.x - 1];
		}
	}

	__syncthreads();

	int i = threadIdx.x;
	int start = 2 * blockDim.x * blockIdx.x;

	if (start + i < numOfElement) {
		inputArray[start + i] += toAdd;
	}

	if (start + i + blockDim.x < numOfElement) {
		inputArray[start + i + blockDim.x] += toAdd;
	}
}

int main() {
	// 4194304 = 2048 * 2048
	int numOfElements = 4194304;
	int* inputArray = new int[numOfElements];
	int* outputArray = new int[numOfElements];

	// Randomly Generate input array
	srand(time(NULL));
	for (int i = 0; i < numOfElements; i++) {
		inputArray[i] = rand() % 3;
	}
	printf("Finished Generating Input Array.\n");

	// Parallel Scan in serial
	int* serialResult = new int[numOfElements];
	for (int i = 0; i < numOfElements; i++) {
		serialResult[i] = inputArray[i];
	}
	clock_t start = clock();
	for (int i = 1; i < numOfElements; i++) {
		serialResult[i] += serialResult[i - 1];
	}
	clock_t end = clock();
	double elapsed = double(end - start) / CLOCKS_PER_SEC;
	printf("Finished Calculating List Scan in Serial. Time Elapsed: %f seconds.\n", elapsed);

	// List Scan in parallel.
	cudaError_t cudaStatus = listScanWithCuda(inputArray, outputArray, numOfElements);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "listScanWithCuda failed!");
		return 1;
	}

	// Check if the parallel result is corret
	bool correct = true;
	for (int i = 0; i < numOfElements; i++) {
		if (abs(outputArray[i] - serialResult[i]) > 0.001) {
			correct = false;
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

	delete(inputArray);
	delete(outputArray);
	delete(serialResult);

	return 0;
}

// List Scan in parallel.
cudaError_t listScanWithCuda(int* inputArray, int* outputArray, int numOfElement) {

	cudaError_t cudaStatus;

	int* dev_in = 0;
	int* dev_out = 0;
	int* dev_blockParitialSumBuffer = 0;
	int* dev_blockScanBuffer = 0;
	int* dev_tmp = 0;

	int numBlocks = ceil((1.0 * numOfElement) / (BLOCK_SIZE * 2));

	// Allocate device memory for input list
	cudaStatus = cudaMalloc((void**)&dev_in, numOfElement * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate device memory for output list
	cudaStatus = cudaMalloc((void**)&dev_out, numOfElement * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate device memory for intermediate blockSumBuffer
	cudaStatus = cudaMalloc((void**)&dev_blockParitialSumBuffer, numBlocks * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate device memory for intermediate dev_blockScanBuffer
	cudaStatus = cudaMalloc((void**)&dev_blockScanBuffer, numBlocks * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate device memory for a temp variable
	cudaStatus = cudaMalloc((void**)&dev_tmp, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input list from host to device.
	cudaStatus = cudaMemcpy(dev_in, inputArray, numOfElement * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// ——————————————————————————————————————————————————————————————————————————————————————————————————
	// Launch Brent-Kung Kernel
	// Each block is parallel scanned after this first call.
	// Each block's partial sum is stored in dev_blockSumBuffer
	dim3 dimGrid(numBlocks, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	listScanKernel_BrentKung<<<dimGrid, dimBlock>>>(dev_in, dev_out, numOfElement, dev_blockParitialSumBuffer);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "listScanKernel_BrentKung launch first time failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after first time launching listScanKernel_BrentKung!\n", cudaStatus);
		goto Error;
	}
	// ——————————————————————————————————————————————————————————————————————————————————————————————————

	// Launch Brent-Kung Kernel
	// Intermediate block partial sum buffer is scanned after this second call
	// All blocks' partial sum is stored in a temp variable, not used further
	listScanKernel_BrentKung<<<(1, 1, 1), dimBlock>>>(dev_blockParitialSumBuffer, dev_blockScanBuffer, numBlocks, dev_tmp);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "listScanKernel_BrentKung launch second time failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after second time launching listScanKernel_BrentKung!\n", cudaStatus);
		goto Error;
	}

	// ——————————————————————————————————————————————————————————————————————————————————————————————————
	addBlockPartialSumScanToEachBlockKernel<<<dimGrid, dimBlock>>>(dev_out, dev_blockScanBuffer, numOfElement);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addBlockPartialSumScanToEachBlockKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addBlockPartialSumScanToEachBlockKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from device to host.
	cudaStatus = cudaMemcpy(outputArray, dev_out, numOfElement * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_blockParitialSumBuffer);
	cudaFree(dev_blockScanBuffer);
	cudaFree(dev_tmp);

	return cudaStatus;
}