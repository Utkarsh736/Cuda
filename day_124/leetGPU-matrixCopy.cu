#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixCopyKernel(const float* A, float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        B[row * N + col] = A[row * N + col];
    }
}

void solve(const float* A, float* B, int N) {
    float *d_A, *d_B;
    size_t size = N * N * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    matrixCopyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}
