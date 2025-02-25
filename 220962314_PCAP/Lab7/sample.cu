#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1024

__global__ void CUDACount(char *A, unsigned int *d_count) {
    int i = threadIdx.x;
    if (A[i] == 'a') {
        atomicAdd(d_count, 1);
    }
}

int main() {
    char A[N];
    char *d_A;
    unsigned int count = 0, result;
    unsigned int *d_count;

    printf("Enter a string: ");
    fgets(A, N, stdin);  // Safe way to read input

    int length = strlen(A);
    if (A[length - 1] == '\n') A[length - 1] = '\0'; // Remove newline character

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate memory
    cudaMalloc((void**)&d_A, length * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(unsigned int));

    // Copy data to device
    cudaMemcpy(d_A, A, length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
    }

    // Launch kernel
    CUDACount<<<1, length>>>(d_A, d_count);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy result back to host
    cudaMemcpy(&result, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Total occurrences of 'a' = %u\n", result);
    printf("Time taken: %f ms\n", elapsedTime);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_count);

    return 0;
}
