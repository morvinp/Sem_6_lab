#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd1(int *A, int *B, int *C, int n) {
    // Kernel 1: n threads in 1 block
    int idx = threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vectorAdd2(int *A, int *B, int *C, int n) {
    // Kernel 2: 1 thread in n blocks
    int idx = blockIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vectorAdd3(int *A, int *B, int *C, int n) {
    // Kernel 3: 256 threads per block, varying number of blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n = 256; // Size of vectors

    // Allocate memory for vectors on host
    int *h_A = (int *)malloc(n * sizeof(int));
    int *h_B = (int *)malloc(n * sizeof(int));
    int *h_C = (int *)malloc(n * sizeof(int));

    // Initialize the vectors
    for (int i = 0; i < n; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate memory for vectors on device
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * sizeof(int));
    cudaMalloc((void**)&d_B, n * sizeof(int));
    cudaMalloc((void**)&d_C, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel 1: n threads in 1 block
    vectorAdd1<<<1, n>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize(); // Ensure the first kernel finishes

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Kernel 1 Result: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    // Re-initialize memory for device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel 2: 1 thread in n blocks
    vectorAdd2<<<n, 1>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize(); // Ensure the second kernel finishes

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Kernel 2 Result: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    // Re-initialize memory for device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel 3: 256 threads per block, varying number of blocks
    dim3 threadsPerBlock(256); // Set the number of threads per block
    dim3 numBlocks((n + 255) / 256); // Calculate the number of blocks needed to handle n elements
    vectorAdd3<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize(); // Ensure the third kernel finishes

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Kernel 3 Result: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}