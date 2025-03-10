#include <cuda_runtime.h>
#include <stdio.h>

// Kernel function to multiply matrices (each row computed by one thread)
__global__ void matrixMultiplyRow(int *A, int *B, int *C, int rowsA, int colsA, int colsB) {
    int row = threadIdx.x;
    
    // Check if row is valid
    if (row < rowsA) {
        // Compute each element of the result row
        for (int col = 0; col < colsB; col++) {
            int sum = 0;
            for (int k = 0; k < colsA; k++) {
                sum += A[row * colsA + k] * B[k * colsB + col];
            }
            C[row * colsB + col] = sum;
        }
    }
}

int main() {
    int rowsA = 3;
    int colsA = 4;
    int rowsB = 4;
    int colsB = 2;
    
    // Verify matrix dimensions are compatible for multiplication
    if (colsA != rowsB) {
        printf("Error: Incompatible matrix dimensions for multiplication.\n");
        return -1;
    }
    
    int sizeA = rowsA * colsA * sizeof(int);
    int sizeB = rowsB * colsB * sizeof(int);
    int sizeC = rowsA * colsB * sizeof(int);
    
    // Host matrices
    int *h_A = (int*)malloc(sizeA);
    int *h_B = (int*)malloc(sizeB);
    int *h_C = (int*)malloc(sizeC);
    
    // Initialize matrices
    for (int i = 0; i < rowsA * colsA; i++) {
        h_A[i] = i + 1;
    }
    
    for (int i = 0; i < rowsB * colsB; i++) {
        h_B[i] = i + 1;
    }
    
    // Print matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            printf("%d ", h_A[i * colsA + j]);
        }
        printf("\n");
    }
    
    // Print matrix B
    printf("Matrix B:\n");
    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            printf("%d ", h_B[i * colsB + j]);
        }
        printf("\n");
    }
    
    // Device matrices
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Launch kernel with one thread per row
    matrixMultiplyRow<<<1, rowsA>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // Print result matrix
    printf("Result Matrix C (A * B):\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            printf("%d ", h_C[i * colsB + j]);
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