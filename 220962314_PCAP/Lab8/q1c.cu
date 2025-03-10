#include <cuda_runtime.h>
#include <stdio.h>

// Kernel function to add two matrices (each element computed by one thread)
__global__ void matrixAddElement(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if row and column are valid
    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

int main() {
    int rows = 4;
    int cols = 5;
    
    int size = rows * cols * sizeof(int);
    
    // Host matrices
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);
    
    // Initialize matrices
    for (int i = 0; i < rows * cols; i++) {
        h_A[i] = i + 1;
        h_B[i] = (i + 1) * 2;
    }
    
    // Print matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_A[i * cols + j]);
        }
        printf("\n");
    }
    
    // Print matrix B
    printf("Matrix B:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_B[i * cols + j]);
        }
        printf("\n");
    }
    
    // Device matrices
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Define thread block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch kernel with one thread per element
    matrixAddElement<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print result matrix
    printf("Result Matrix C (A + B):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_C[i * cols + j]);
        }
        printf("\n");
    }
    
    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}