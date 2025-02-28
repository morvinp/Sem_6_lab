#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void odd_even_step(int *da, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1 || i % 2 == 0) return; 
    
    if (da[i] > da[i + 1]) {
        int temp = da[i];
        da[i] = da[i + 1];
        da[i + 1] = temp;
    }
}

__global__ void even_odd_step(int *da, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1 || i % 2 == 1) return; 
    
    if (da[i] > da[i + 1]) {
        int temp = da[i];
        da[i] = da[i + 1];
        da[i + 1] = temp;
    }
}

int main() {
    int n;
    printf("Enter the length of the vector: ");
    scanf("%d", &n);

    int *a = (int*)malloc(n * sizeof(int));
    int *c = (int*)malloc(n * sizeof(int));
    int *da, *dc;

    cudaMalloc((void**)&da, n * sizeof(int));
    cudaMalloc((void**)&dc, n * sizeof(int));

    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }

    cudaMemcpy(da, a, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Perform multiple phases to ensure sorting
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            // Odd phase (Odd-Even Sort)
            odd_even_step<<<numBlocks, blockSize>>>(da, n);
        } else {
            // Even phase (Even-Odd Sort)
            even_odd_step<<<numBlocks, blockSize>>>(da, n);
        }
        cudaDeviceSynchronize();  // Ensure all threads are done before starting the next phase
    }

    cudaMemcpy(c, da, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    cudaFree(da);
    cudaFree(dc);
    free(a);
    free(c);

    return 0;
}
