// spmv.cu
#include <cuda_runtime.h>
#include <cstdio>

// CUDA kernel for SpMV using CSR format
__global__ void spmv_csr_kernel(int M, const int* row_ptr, const int* col_idx,
                                const float* values, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float dot = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int i = row_start; i < row_end; ++i) {
            dot += values[i] * x[col_idx[i]];
        }

        y[row] = dot;
    }
}

// Host function: SpMV using CSR
extern "C" void solve(int M, int N, int nnz,
                      const int* h_row_ptr, const int* h_col_idx, const float* h_values,
                      const float* h_x, float* h_y) {
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    spmv_csr_kernel<<<blocks, threads>>>(M, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

// Optional test main
#ifdef TEST_SPMV
int main() {
    // Example matrix:
    // [10 0  0]
    // [0  20 0]
    // [30 0  40]
    int M = 3, N = 3, nnz = 5;
    int row_ptr[] = {0, 1, 2, 5};
    int col_idx[] = {0, 1, 0, 2, 2};
    float values[] = {10, 20, 30, 0, 40};
    float x[] = {1, 2, 3};
    float y[3];

    solve(M, N, nnz, row_ptr, col_idx, values, x, y);

    printf("Result y = [");
    for (int i = 0; i < M; ++i) printf("%.1f ", y[i]);
    printf("]\n");
    return 0;
}
#endif
