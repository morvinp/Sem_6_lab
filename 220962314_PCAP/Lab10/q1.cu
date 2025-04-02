#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  // Define matrix size (N x N)

// CUDA kernel to perform matrix multiplication
__global__ void matrixMul(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

void printMatrix(int *matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int a[N * N], b[N * N], c[N * N];  // Host matrices
    int *d_a, *d_b, *d_c;              // Device matrices
    int size = N * N * sizeof(int);

    // Initialize input matrices
    for (int i = 0; i < N * N; ++i) {
        a[i] = 1;  // Initializing matrix A with all 1s
        b[i] = 2;  // Initializing matrix B with all 2s
    }

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define block and grid size (2D grid and 2D block)
    dim3 blockDim(2, 2);  // Block size: 2x2 threads
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Matrix A:\n");
    printMatrix(a, N);
    printf("\nMatrix B:\n");
    printMatrix(b, N);
    printf("\nResultant Matrix C (A x B):\n");
    printMatrix(c, N);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
