#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  // Matrix size (4x4)
#define NNZ 7  // Number of non-zero elements

// Kernel for SpMV using CSR format
__global__ void spmv_csr(int *rowPtr, int *colIdx, float *values, float *x, float *y, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0;
        for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
            sum += values[i] * x[colIdx[i]];
        }
        y[row] = sum;
    }
}

int main() {
    // Host data: CSR representation of a sparse matrix
    int h_rowPtr[N+1] = {0, 2, 4, 5, 7};  // Row pointers
    int h_colIdx[NNZ] = {0, 1, 1, 2, 2, 0, 3};  // Column indices
    float h_values[NNZ] = {1, 2, 3, 4, 5, 6, 7};  // Non-zero values
    float h_x[N] = {1, 2, 3, 4};  // Input vector
    float h_y[N] = {0, 0, 0, 0};  // Output vector

    // Device memory allocation
    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (N+1) * sizeof(int));
    cudaMalloc(&d_colIdx, NNZ * sizeof(int));
    cudaMalloc(&d_values, NNZ * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_rowPtr, h_rowPtr, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, NNZ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch: 1 thread per row
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr<<<blocksPerGrid, threadsPerBlock>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, N);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result (y = A * x):\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_y[i]);
    }

    // Free memory
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
