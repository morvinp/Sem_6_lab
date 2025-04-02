#include <stdio.h>
#include <cuda_runtime.h>

#define N 10  // Array size (can be modified for testing)
#define BLOCK_SIZE 256  // Block size for parallel computation

// CUDA kernel for inclusive scan (prefix sum) using parallel reduction
__global__ void inclusiveScan(int *d_in, int *d_out, int n) {
    __shared__ int temp[BLOCK_SIZE];  // Shared memory for block

    int tid = threadIdx.x;  // Local thread ID
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID

    // Load data into shared memory
    if (global_tid < n) {
        temp[tid] = d_in[global_tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Perform inclusive scan using parallel reduction
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write results back to global memory
    if (global_tid < n) {
        d_out[global_tid] = temp[tid];
    }
}

// Host function to initialize input array
void initializeArray(int *arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = 1;  // Initializing with 1 for simplicity
    }
}

// Host function to print array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int h_in[N], h_out[N];  // Host arrays
    int *d_in, *d_out;      // Device arrays
    int size = N * sizeof(int);

    // Initialize input array
    initializeArray(h_in, N);

    // Allocate device memory
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // Copy input array from host to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch inclusive scan kernel
    inclusiveScan<<<gridDim, blockDim>>>(d_in, d_out, N);

    // Copy results from device to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Input Array:\n");
    printArray(h_in, N);
    printf("\nInclusive Scan Result:\n");
    printArray(h_out, N);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
