#include <stdio.h>
#include <cuda_runtime.h>

#define N 4       // Number of rows in matrix
#define NNZ 7     // Number of non-zero elements
#define BLOCK_SIZE 4  // Set block size to match row count for simplicity

// Kernel for SpMV using CSR with shared memory optimization
__global__ void spmv_csr_shared(int *rowPtr, int *colIdx, float *values, float *x, float *y, int rows) {
    __shared__ float x_shared[N];  // Shared memory for input vector

    int thread_id = threadIdx.x;
    int row = blockIdx.x * blockDim.x + thread_id;

    // Load input vector `x` into shared memory
    if (thread_id < N) {
        x_shared[thread_id] = x[thread_id];
    }
    __syncthreads();

    if (row < rows) {
        float sum = 0.0;
        for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
            sum += values[i] * x_shared[colIdx[i]];
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

    // Kernel launch with shared memory
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr_shared<<<blocksPerGrid, threadsPerBlock>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, N);

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
