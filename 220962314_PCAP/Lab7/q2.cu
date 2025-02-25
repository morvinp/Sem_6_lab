#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h>

#define MAX_LEN 256  // Maximum length of input string

__global__ void transformString(char *S, char *RS, int *pos, int len) {
    int tid = threadIdx.x;  // Each thread handles one character of S

    if (tid < len) {
        int start = pos[tid];  // Starting position in RS
        for (int j = 0; j <= tid; j++) {  // Copy (i+1) times
            RS[start + j] = S[tid];
        }
    }
}

int main() {
    char S[MAX_LEN], *d_S, *d_RS;
    int *d_pos;
    char RS[MAX_LEN * MAX_LEN];  // Large enough for worst case
    int pos[MAX_LEN];  // Stores start index for each character in RS

    printf("Enter string S: ");
    scanf("%s", S);

    int len = strlen(S);
    int totalLen = 0;

    // Compute starting indices in RS
    for (int i = 0; i < len; i++) {
        pos[i] = totalLen;
        totalLen += (i + 1);
    }

    // Allocate memory on GPU
    cudaMalloc((void**)&d_S, len * sizeof(char));
    cudaMalloc((void**)&d_RS, totalLen * sizeof(char));
    cudaMalloc((void**)&d_pos, len * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_S, S, len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos, len * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with one block and len threads
    transformString<<<1, len>>>(d_S, d_RS, d_pos, len);

    // Copy result back
    cudaMemcpy(RS, d_RS, totalLen * sizeof(char), cudaMemcpyDeviceToHost);

    RS[totalLen] = '\0';  // Null-terminate string

    printf("Output string RS: %s\n", RS);

    // Free GPU memory
    cudaFree(d_S);
    cudaFree(d_RS);
    cudaFree(d_pos);

    return 0;
}
