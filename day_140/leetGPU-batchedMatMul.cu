// batched_matmul.cu
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel: Each thread computes one element C[b][i][j]
__global__ void batched_matmul_kernel(const float* A, const float* B, float* C,
                                      int BATCH, int M, int K, int N) {
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= BATCH || i >= M || j >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[b * M * K + i * K + k];
        float b_val = B[b * K * N + k * N + j];
        sum += a * b_val;
    }

    C[b * M * N + i * N + j] = sum;
}

extern "C" void solve(const float* h_A, const float* h_B, float* h_C,
                      int BATCH, int M, int K, int N) {
    size_t sizeA = BATCH * M * K * sizeof(float);
    size_t sizeB = BATCH * K * N * sizeof(float);
    size_t sizeC = BATCH * M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16, BATCH);

    batched_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, BATCH, M, K, N);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
