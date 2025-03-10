#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul(int *A, int *B, int *C, int wb, int wa) {
    int rid = threadIdx.x;
    int sum;
    for(int cid = 0; cid < wb; cid++) {
        sum = 0;
        for(int k = 0; k < wa; k++) {
            sum += A[rid*wa+k] * B[cid+wb*k];
        }
        C[rid*wb+cid] = sum;
    }
}

int main() {
    int rowsA = 3;
    int colsA = 2;
    int colsB = 4;
    
    int sizeA = rowsA * colsA * sizeof(int);
    int sizeB = colsA * colsB * sizeof(int);
    int sizeC = rowsA * colsB * sizeof(int);
    
    int *h_A = (int*)malloc(sizeA);
    int *h_B = (int*)malloc(sizeB);
    int *h_C = (int*)malloc(sizeC);
    
    for (int i = 0; i < rowsA * colsA; i++) h_A[i] = i + 1;
    for (int i = 0; i < colsA * colsB; i++) h_B[i] = i + 1;
    
    printf("Matrix A:\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            printf("%d ", h_A[i*colsA + j]);
        }
        printf("\n");
    }
    
    printf("Matrix B:\n");
    for (int i = 0; i < colsA; i++) {
        for (int j = 0; j < colsB; j++) {
            printf("%d ", h_B[j + i*colsB]);
        }
        printf("\n");
    }
    
    // Device matrices
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    matmul<<<1, rowsA>>>(d_A, d_B, d_C, colsB, colsA);
    
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    printf("Result Matrix C:\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            printf("%d ", h_C[i*colsB + j]);
        }
        printf("\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}