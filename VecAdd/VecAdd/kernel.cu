#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <iostream>
#include <cstdlib>

void vecAdd(float* A, float* B, float* C, int n);

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n) {
	cudaError_t err = cudaSuccess;
	int size = n * sizeof(float);

	float* d_A = NULL;
	float* d_B = NULL;
	float* d_C = NULL;

	// Allocate device memory for array A
	err = cudaMalloc((void**)&d_A, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate device memory for array B
	err = cudaMalloc((void**)&d_B, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate device memory for array C
	err = cudaMalloc((void**)&d_C, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy input arrays A and B to device for calculation
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch Kernel, define grid size and block size
	int blockSize = 256;
	int gridSize = ceil(n / 256.0);
	vecAddKernel <<<gridSize, blockSize>>> (d_A, d_B, d_C, n);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy result back to host memory
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr,
			"Failed to copy vector C from device to host (error code %s)!\n",
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free device global memory
	err = cudaFree(d_A);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


int main(void) {
	int n = 102400;
	printf("Adding %d elements\n", n);

	// Allocate host memory for input output and test arrays
	float* h_A = (float*)malloc(sizeof(float) * n);
	float* h_B = (float*)malloc(sizeof(float) * n);
	float* h_C = (float*)malloc(sizeof(float) * n);
	float* answer_C = (float*)malloc(sizeof(float) * n);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL || answer_C == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input arrays
	for (int i = 0; i < n; i++) {
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	// calculate answer C serially
	for (int i = 0; i < n; i++) {
		answer_C[i] = h_A[i] + h_B[i];
	}

	// Prepare for parallel computing
	vecAdd(h_A, h_B, h_C, n);

	// Test if the answer is correct
	for (int i = 0; i < n; i++) {
		if (abs(h_C[i] - answer_C[i]) > 1e-4) {
			printf("Something wrong\n");
		}
	}

	printf("Everything is great.\n");

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Finished.\n");
	return 0;
}