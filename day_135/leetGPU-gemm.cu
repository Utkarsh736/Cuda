// gemm_fp16.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// CUDA kernel: C = alpha * A * B + beta * C
__global__ void gemm_fp16_kernel(const __half* A, const __half* B, __half* C,
                                 int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // [0, M)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // [0, N)

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a = __half2float(A[row * K + i]);
            float b = __half2float(B[i * N + col]);
            sum += a * b;
        }

        float c_old = __half2float(C[row * N + col]);
        float c_new = alpha * sum + beta * c_old;
        C[row * N + col] = __float2half(c_new);
    }
}

// Host-side solve function
extern "C" void solve(const __half* A, const __half* B, __half* C,
                      int M, int N, int K, float alpha, float beta) {
    __half *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));

    cudaMemcpy(d_A, A, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    gemm_fp16_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaMemcpy(C, d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
