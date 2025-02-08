#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void vectorAdd1(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x;  // Global thread index is directly the thread index
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  // Add corresponding elements
    }
}

__global__ void vectorAdd2(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x;  // Global thread index is the block index
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  // Add corresponding elements
    }
}

int main(void) {
    int N = 1000;  // Length of the vectors (can be changed)
    int size = N * sizeof(int);  // Size of each vector in bytes

    int *h_a = (int *)malloc(size);  // Host memory for vector A
    int *h_b = (int *)malloc(size);  // Host memory for vector B
    int *h_c1 = (int *)malloc(size);  // Host memory for result vector C (kernel 1)
    int *h_c2 = (int *)malloc(size);  // Host memory for result vector C (kernel 2)

    int *d_a, *d_b, *d_c1, *d_c2;  // Device pointers for vectors

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i + 1;  // 1, 2, 3, 4, ...
        h_b[i] = (i + 1) * 2;  // 2, 4, 6, 8, ...
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c1, size);
    cudaMalloc((void **)&d_c2, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch the first kernel with 1 block and N threads (1, N)
    vectorAdd1<<<1, N>>>(d_a, d_b, d_c1, N);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error (kernel 1): %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the first kernel to finish
    cudaDeviceSynchronize();

    // Launch the second kernel with N blocks and 1 thread (N, 1)
    vectorAdd2<<<N, 1>>>(d_a, d_b, d_c2, N);

    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error (kernel 2): %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the second kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to host memory
    cudaMemcpy(h_c1, d_c1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c2, d_c2, size, cudaMemcpyDeviceToHost);

    // Print the result of the first kernel (optional, printing first 10 elements)
    printf("Result from kernel 1 (1, N): ");
    for (int i = 0; i < (N < 10 ? N : 10); i++) {
        printf("%d ", h_c1[i]);
    }
    printf("\n");

    // Print the result of the second kernel (optional, printing first 10 elements)
    printf("Result from kernel 2 (N, 1): ");
    for (int i = 0; i < (N < 10 ? N : 10); i++) {
        printf("%d ", h_c2[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c1);
    cudaFree(d_c2);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c1);
    free(h_c2);

    return 0;
}
