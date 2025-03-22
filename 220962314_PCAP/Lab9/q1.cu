#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

void convertToCSR(float **matrix, int rows, int cols, int **row_ptr, int **col_index, float **data, int *nnz) {
    int count = 0;
    *row_ptr = (int *)malloc((rows + 1) * sizeof(int));
    *row_ptr[0] = 0;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] != 0) {
                count++;
            }
        }
        (*row_ptr)[i + 1] = count;
    }

    *col_index = (int *)malloc(count * sizeof(int));
    *data = (float *)malloc(count * sizeof(float));
    
    int k = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] != 0) {
                (*data)[k] = matrix[i][j];
                (*col_index)[k] = j;
                k++;
            }
        }
    }
    *nnz = count;
}

int main() {
    int rows, cols;
    printf("Enter matrix dimensions (rows cols): ");
    scanf("%d %d", &rows, &cols);

    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float *)malloc(cols * sizeof(float));
    }
    
    printf("Enter matrix values row-wise:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%f", &matrix[i][j]);
        }
    }
    
    int *h_row_ptr, *h_col_index;
    float *h_data;
    int nnz;
    convertToCSR(matrix, rows, cols, &h_row_ptr, &h_col_index, &h_data, &nnz);
    
    float *h_x = (float *)malloc(cols * sizeof(float));
    float *h_y = (float *)calloc(rows, sizeof(float));
    float *h_y_initial = (float *)malloc(rows * sizeof(float));

    printf("Enter vector x: ");
    for (int i = 0; i < cols; i++) {
        scanf("%f", &h_x[i]);
    }
    
    printf("Enter initial vector y: ");
    for (int i = 0; i < rows; i++) {
        scanf("%f", &h_y_initial[i]);
        h_y[i] = h_y_initial[i];
    }

    int *d_row_ptr, *d_col_index;
    float *d_data, *d_x, *d_y;
    cudaMalloc((void **)&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_index, nnz * sizeof(int));
    cudaMalloc((void **)&d_data, nnz * sizeof(float));
    cudaMalloc((void **)&d_x, cols * sizeof(float));
    cudaMalloc((void **)&d_y, rows * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index, h_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, rows * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr<<<blocksPerGrid, threadsPerBlock>>>(d_row_ptr, d_col_index, d_data, d_x, d_y, rows);
    
    cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resulting vector y: ");
    for (int i = 0; i < rows; i++) {
        printf("%f ", h_y[i]);
    }
    printf("\n");

    cudaFree(d_row_ptr);
    cudaFree(d_col_index);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_row_ptr);
    free(h_col_index);
    free(h_data);
    free(h_x);
    free(h_y);
    free(h_y_initial);

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);

    return 0;
}