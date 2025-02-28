#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void odd_even_step(int *da, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;  // Ensure no out-of-bounds access
    
    // Odd step: compare elements at odd indices
    if (i % 2 == 1 && i < n - 1) {
        if (da[i] > da[i + 1]) {
            // Swap if out of order
            int temp = da[i];
            da[i] = da[i + 1];
            da[i + 1] = temp;
        }
    }
}

__global__ void even_odd_step(int *da, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;  // Ensure no out-of-bounds access
    
    // Even step: compare elements at even indices
    if (i % 2 == 0) {
        if (da[i] > da[i + 1]) {
            // Swap if out of order
            int temp = da[i];
            da[i] = da[i + 1];
            da[i + 1] = temp;
        }
    }
}

int main() {
    int n;
    printf("Enter the length of the vector: ");
    scanf("%d", &n);

    int *a = (int*)malloc(n * sizeof(int));
    int *c = (int*)malloc(n * sizeof(int));
    int *da;

    cudaMalloc((void**)&da, n * sizeof(int));

    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }

    cudaMemcpy(da, a, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Perform multiple phases to ensure sorting
    for (int phase = 0; phase < n; phase++) {
        // Perform Odd-Even Step
        odd_even_step<<<numBlocks, blockSize>>>(da, n);
        cudaDeviceSynchronize();  // Ensure all threads finish before the next step
        
        // Perform Even-Odd Step
        even_odd_step<<<numBlocks, blockSize>>>(da, n);
        cudaDeviceSynchronize();  // Ensure all threads finish before the next step
    }

    cudaMemcpy(c, da, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    cudaFree(da);
    free(a);
    free(c);

    return 0;
}
