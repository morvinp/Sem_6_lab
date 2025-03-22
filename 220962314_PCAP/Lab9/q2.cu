#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void transformMatrix(float *d_mat, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int index = row * N + col;
        float val = d_mat[index];
        for (int i = 1; i < row + 1; i++) { 
            d_mat[index] *= val;  // Multiply iteratively to raise power
        }
    }
}

int main() {
    int M, N;
    printf("Enter matrix dimensions (M N): ");
    scanf("%d %d", &M, &N);

    float *h_mat = (float *)malloc(M * N * sizeof(float));

    printf("Enter matrix elements row-wise:\n");
    for (int i = 0; i < M * N; i++) {
        scanf("%f", &h_mat[i]);
    }

    float *d_mat;
    cudaMalloc(&d_mat, M * N * sizeof(float));
    cudaMemcpy(d_mat, h_mat, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transformMatrix<<<numBlocks, threadsPerBlock>>>(d_mat, M, N);

    cudaMemcpy(h_mat, d_mat, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Modified matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_mat[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_mat);
    free(h_mat);
    return 0;
}