#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void selsort(int *da, int *dc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // Ensure we don't go out of bounds

    int k = 0;
    for (int j = 0; j < n; j++) {
        if ((da[j] < da[i]) || (da[j] == da[i] && j < i))
            k++;
    }
    dc[k] = da[i];  // Store sorted value at correct position
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
    selsort<<<numBlocks, blockSize>>>(da, dc, n);

    cudaMemcpy(c, dc, n * sizeof(int), cudaMemcpyDeviceToHost);

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
 
