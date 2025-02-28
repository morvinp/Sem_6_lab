#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void Convolution(float* da, float* db, float* dc, int maskWidth, int vectorSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectorSize) return; 

    int s = i - maskWidth / 2;
    float sum = 0;
    
    for (int j = 0; j < maskWidth; j++) {
        if (s + j >= 0 && s + j < vectorSize) {
            sum += da[s + j] * db[j];
        }
    }
    dc[i] = sum;
}

int main() {
    int n1, n2;

    printf("Length of the vector: ");
    scanf("%d", &n1);

    printf("Enter the length of the mask: ");
    scanf("%d", &n2);

    
    float* a = (float*)malloc(n1 * sizeof(float));
    float* b = (float*)malloc(n2 * sizeof(float));
    float* c = (float*)malloc(n1 * sizeof(float));

    float *da, *db, *dc;
    cudaMalloc((void**)&da, n1 * sizeof(float));
    cudaMalloc((void**)&db, n2 * sizeof(float));
    cudaMalloc((void**)&dc, n1 * sizeof(float));

    printf("Enter vector elements: ");
    for (int i = 0; i < n1; i++)
        scanf("%f", &a[i]);

    printf("Enter mask elements: ");
    for (int i = 0; i < n2; i++)
        scanf("%f", &b[i]);

    cudaMemcpy(da, a, n1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n2 * sizeof(float), cudaMemcpyHostToDevice);

   
    int threadsPerBlock = 256;
    int blocksPerGrid = (n1 + threadsPerBlock - 1) / threadsPerBlock;

    Convolution<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, n2, n1);
    cudaDeviceSynchronize(); 

    cudaMemcpy(c, dc, n1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Convolution Result:\n");
    for (int i = 0; i < n1; i++)
        printf("%f\t", c[i]);
    printf("\n");


    free(a);
    free(b);
    free(c);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}

