#include <cuda_runtime.h>
#include <iostream>
#include <bitset>
using namespace std;

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
string toBinaryTrimmed(int num) {
    if (num == 0) return "0"; // Special case for zero
    string binary = bitset<8>(num).to_string();  // Convert to 8-bit binary
    return binary.substr(binary.find('1')); // Trim leading zeros
}

int main() {
    int M, N;
    cout << "Enter matrix dimensions (M N): ";
    cin >> M >> N;

    int *h_A = new int[M * N];
    int *h_B = new int[M * N];

    cout << "Enter matrix elements row-wise:" << endl;
    for (int i = 0; i < M * N; i++) {
        cin >> h_A[i];
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

    cout << "Modified matrix B (in binary format):" << endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i > 0 && i < M - 1 && j > 0 && j < N - 1)
                cout << toBinaryTrimmed(h_B[i * N + j]) << " ";  // Print binary without leading zeros
            else
                cout << h_B[i * N + j] << " ";  // Print normal numbers
        }
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;
    
    return 0;
}
