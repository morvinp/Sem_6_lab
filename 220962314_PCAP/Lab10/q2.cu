#include <stdio.h>
#include <cuda_runtime.h>

#define N 10       // Input size
#define K 5             // Kernel size
#define BLOCK_SIZE 256  // Block size for parallelism

// Constant memory for kernel
__constant__ int d_kernel[K];

// CUDA kernel to perform 1D convolution using constant memory
__global__ void conv1D(int *d_input, int *d_output, int n, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID
    int radius = k / 2;  // Radius of the kernel (for symmetric kernel)

    int sum = 0;
    if (tid < n) {
        // Apply convolution by accessing the constant memory for kernel
        for (int j = -radius; j <= radius; ++j) {
            int idx = tid + j;
            if (idx >= 0 && idx < n) {  // Boundary check
                sum += d_input[idx] * d_kernel[j + radius];
            }
        }
        d_output[tid] = sum;
    }
}

// Function to initialize input and kernel data
void initializeData(int *input, int *kernel, int n, int k) {
    for (int i = 0; i < n; ++i) {
        input[i] = rand() % 10;  // Random values for input
    }
    for (int i = 0; i < k; ++i) {
        kernel[i] = 1;  // Uniform kernel with all 1s for simplicity
    }
}

// Function to print the array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int h_input[N], h_kernel[K], h_output[N];  // Host memory
    int *d_input, *d_output;                   // Device memory
    int size_input = N * sizeof(int);
    int size_output = N * sizeof(int);

    // Initialize data
    initializeData(h_input, h_kernel, N, K);

    // Allocate memory on GPU
    cudaMalloc((void **)&d_input, size_input);
    cudaMalloc((void **)&d_output, size_output);

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, h_kernel, K * sizeof(int));

    // Define block and grid sizes
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(gridSize);

    // Launch convolution kernel
    conv1D<<<gridDim, blockDim>>>(d_input, d_output, N, K);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Print results
    printf("Input Array:\n");
    printArray(h_input, N);
    printf("\nKernel Array:\n");
    printArray(h_kernel, K);
    printf("\nResult of 1D Convolution:\n");
    printArray(h_output, N);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
