
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <random>

#define BLOCK_SIZE 32

cudaError_t simpleMatrixMultiplicationWithCUDA(float* matrixA, float* matrixB, float* matrixC,
											   int matrixARows, int matrixACols,
											   int matrixBRows, int matrixBCols,
											   int matrixCRows, int matrixCCols);
void getSerialMatrixC(float* matrixA, float* matrixB, float* matrixC,
					  int matrixARows, int matrixACols,
					  int matrixBRows, int matrixBCols,
					  int matrixCRows, int matrixCCols);
void printMatrix(float* matrix, int matrixRows, int matrixCols);

__global__ void simpleMatrixMultiplicationKernel(float* matrixA, float* matrixB, float* matrixC,
												 int matrixARows, int matrixACols,
												 int matrixBRows, int matrixBCols,
												 int matrixCRows, int matrixCCols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < matrixCRows && col < matrixCCols) {
		float product = 0.0f;
		for (int k = 0; k < matrixACols; k++) {
			product += matrixA[row * matrixACols + k] * matrixB[k * matrixBCols + col];
		}
		matrixC[row * matrixCCols + col] = product;
	}
}

int main() {
	int matrixARows = 2291;
	int matrixACols = 3738;
	int matrixBRows = 3738;
	int matrixBCols = 1372;
	int matrixCRows = 2291;
	int matrixCCols = 1372;

	float* matrixA = new float[matrixARows * matrixACols];
	float* matrixB = new float[matrixBRows * matrixBCols];
	float* matrixC = new float[matrixCRows * matrixCCols];
	float* serialMatrixC = new float[matrixCRows * matrixCCols];

	srand(time(NULL));
	for (int i = 0; i < matrixARows; i++) {
		for (int j = 0; j < matrixACols; j++) {
			matrixA[i * matrixACols + j] = (float(rand()) / float((RAND_MAX)) * 10.0);
		}
	}
	printf("Finished generating matrix A.\n");
	//printMatrix(matrixA, matrixARows, matrixACols);

	for (int i = 0; i < matrixBRows; i++) {
		for (int j = 0; j < matrixBCols; j++) {
			matrixB[i * matrixBCols + j] = (float(rand()) / float((RAND_MAX)) * 10.0);
		}
	}
	printf("Finished generating matrix B.\n");
	//printMatrix(matrixB, matrixBRows, matrixBCols);

	// Use Serialized method to generate answer
	clock_t start = clock();
	getSerialMatrixC(matrixA, matrixB, serialMatrixC,
					 matrixARows, matrixACols,
					 matrixBRows, matrixBCols,
					 matrixCRows, matrixCCols);
	clock_t end = clock();
	double time_taken = ((double)end - start) / CLOCKS_PER_SEC;
	printf("Finished generating matrix C by serial method. Time elipsed: %f s\n", time_taken);
	//printMatrix(serialMatrixC, matrixCRows, matrixCCols);

	// Use Parallel method to generate answer
	cudaError_t cudaStatus = simpleMatrixMultiplicationWithCUDA(matrixA, matrixB, matrixC,
																matrixARows, matrixACols,
																matrixBRows, matrixBCols,
																matrixCRows, matrixCCols);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "simpleMatrixMultiplicationWithCUDA failed!");
		return 1;
	}
	printf("Finished generating matrix C by parallel method.\n");
	//printMatrix(matrixC, matrixCRows, matrixCCols);

	// Check if the result is correct
	bool correct = true;
	for (int i = 0; i < matrixCRows; i++) {
		for (int j = 0; j < matrixCCols; j++) {
			if (abs(matrixC[i * matrixCCols + j] - serialMatrixC[i * matrixCCols + j]) > 0.001) {
				correct = false;
			}
		}
	}
	if (correct) {
		printf("All Correct!\n");
	} else {
		printf("Something Wrong!\n");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	delete(matrixA);
	delete(matrixB);
	delete(matrixC);
	delete(serialMatrixC);
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t simpleMatrixMultiplicationWithCUDA(float* matrixA, float* matrixB, float* matrixC,
											   int matrixARows, int matrixACols,
											   int matrixBRows, int matrixBCols,
											   int matrixCRows, int matrixCCols) {
	float* dev_a = 0;
	float* dev_b = 0;
	float* dev_c = 0;
	cudaError_t cudaStatus;

	// Allocate memory for Matrix A
	cudaStatus = cudaMalloc((void**)&dev_a, matrixARows * matrixACols * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate memory for Matrix B
	cudaStatus = cudaMalloc((void**)&dev_b, matrixBRows * matrixBCols * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate memory for Matrix C
	cudaStatus = cudaMalloc((void**)&dev_c, matrixCRows * matrixCCols * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy matrix A from host to device.
	cudaStatus = cudaMemcpy(dev_a, matrixA, matrixARows * matrixACols * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy matrix B from host to device.
	cudaStatus = cudaMemcpy(dev_b, matrixB, matrixBRows * matrixBCols * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimGrid(ceil(1.0 * matrixCCols / BLOCK_SIZE), ceil(1.0 * matrixCRows / BLOCK_SIZE), 1);
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	simpleMatrixMultiplicationKernel <<<dimGrid, dimBlock>>> (dev_a, dev_b, dev_c,
																 matrixARows, matrixACols,
																 matrixBRows, matrixBCols,
																 matrixCRows, matrixCCols);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Parallel Matrix Multiplication Elapsed time : %f ms\n", elapsedTime);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "simpleMatrixMultiplicationKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simpleMatrixMultiplicationKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(matrixC, dev_c, matrixCRows * matrixCCols * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return cudaStatus;
}

void getSerialMatrixC(float* matrixA, float* matrixB, float* matrixC,
					  int matrixARows, int matrixACols,
					  int matrixBRows, int matrixBCols,
					  int matrixCRows, int matrixCCols) {
	for (int i = 0; i < matrixCRows; i++) {
		for (int j = 0; j < matrixCCols; j++) {
			float product = 0.0f;
			for (int k = 0; k < matrixACols; k++) {
				product += matrixA[i * matrixACols + k] * matrixB[k * matrixBCols + j];
			}
			matrixC[i * matrixCCols + j] = product;
		}
	}
}

void printMatrix(float* matrix, int matrixRows, int matrixCols) {
	for (int i = 0; i < matrixRows; i++) {
		for (int j = 0; j < matrixCols; j++) {
			printf("%f ", matrix[i + matrixCols + j]);
		}
		printf("\n");
	}
}