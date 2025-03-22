#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel to compute 1's complement for non-border elements
__global__ void transformMatrix(int *d_A, int *d_B, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int index = row * N + col;
        if (row > 0 && row < M - 1 && col > 0 && col < N - 1) {
            int num = d_A[index];
            int onesComplement = 0, power = 1;
            while (num > 0) {
                int bit = num % 2;
                onesComplement += (1 - bit) * power;  // Flip the bit
                power *= 2;
                num /= 2;
            }
            d_B[index] = onesComplement;
        } else {
            d_B[index] = d_A[index];  // Keep border elements same
        }
    }
}

// Function to convert a decimal number to trimmed binary string
void toBinaryTrimmed(int num, char *binaryStr) {
    if (num == 0) {
        sprintf(binaryStr, "0");
        return;
    }
    char temp[33];
    int index = 0;
    while (num > 0) {
        temp[index++] = (num % 2) + '0';
        num /= 2;
    }
    temp[index] = '\0';
    for (int i = 0; i < index; i++) {
        binaryStr[i] = temp[index - 1 - i];
    }
    binaryStr[index] = '\0';
}

int main() {
    int M, N;
    printf("Enter matrix dimensions (M N): ");
    scanf("%d %d", &M, &N);

    int *h_A = (int *)malloc(M * N * sizeof(int));
    int *h_B = (int *)malloc(M * N * sizeof(int));

    printf("Enter matrix elements row-wise:\n");
    for (int i = 0; i < M * N; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A, *d_B;
    cudaMalloc(&d_A, M * N * sizeof(int));
    cudaMalloc(&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transformMatrix<<<numBlocks, threadsPerBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Modified matrix B (in binary format):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i > 0 && i < M - 1 && j > 0 && j < N - 1) {
                char binaryStr[33];
                toBinaryTrimmed(h_B[i * N + j], binaryStr);
                printf("%s ", binaryStr);
            } else {
                printf("%d ", h_B[i * N + j]);
            }
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    
    return 0;
}
