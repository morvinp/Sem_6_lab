#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA kernel for Sparse Matrix-Vector Multiplication (SpMV) using CSR format
__global__ void spmv_csr(int *row_ptr, int *col_index, float *data, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int ele = row_start; ele < row_end; ele++) {
            dot += data[ele] * x[col_index[ele]];
        }
        y[row] += dot; // Adding to existing y vector
    }
}

void convertToCSR(const std::vector<std::vector<float>> &matrix, std::vector<int> &row_ptr, std::vector<int> &col_index, std::vector<float> &data) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    row_ptr.push_back(0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] != 0) {
                data.push_back(matrix[i][j]);
                col_index.push_back(j);
            }
        }
        row_ptr.push_back(data.size());
    }
}

int main() {
    int rows, cols;
    std::cout << "Enter matrix dimensions (rows cols): ";
    std::cin >> rows >> cols;
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    
    std::cout << "Enter matrix values row-wise:" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cin >> matrix[i][j];
        }
    }
    
    std::vector<int> h_row_ptr, h_col_index;
    std::vector<float> h_data;
    convertToCSR(matrix, h_row_ptr, h_col_index, h_data);
    
    std::vector<float> h_x(cols), h_y(rows, 0), h_y_initial(rows);
    std::cout << "Enter vector x: ";
    for (int i = 0; i < cols; i++) {
        std::cin >> h_x[i];
    }
    
    std::cout << "Enter initial vector y: ";
    for (int i = 0; i < rows; i++) {
        std::cin >> h_y_initial[i];
        h_y[i] = h_y_initial[i]; // Copy initial y values
    }

    int *d_row_ptr, *d_col_index;
    float *d_data, *d_x, *d_y;
    cudaMalloc((void **)&d_row_ptr, h_row_ptr.size() * sizeof(int));
    cudaMalloc((void **)&d_col_index, h_col_index.size() * sizeof(int));
    cudaMalloc((void **)&d_data, h_data.size() * sizeof(float));
    cudaMalloc((void **)&d_x, cols * sizeof(float));
    cudaMalloc((void **)&d_y, rows * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), h_row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index, h_col_index.data(), h_col_index.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), rows * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr<<<blocksPerGrid, threadsPerBlock>>>(d_row_ptr, d_col_index, d_data, d_x, d_y, rows);
    
    cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Resulting vector y: ";
    for (int i = 0; i < rows; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_row_ptr);
    cudaFree(d_col_index);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
