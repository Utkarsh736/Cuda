// quantized_matmul_int8.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

__global__ void quantized_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C,
                                        int M, int N, int K,
                                        float sA, float sB, float sC,
                                        int zA, int zB, int zC) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N

    if (row < M && col < N) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k) {
            int32_t a_val = static_cast<int32_t>(A[row * K + k]) - zA;
            int32_t b_val = static_cast<int32_t>(B[k * N + col]) - zB;
            acc += a_val * b_val;
        }

        float scaled = acc * (sA * sB / sC);
        int32_t rounded = static_cast<int32_t>(roundf(scaled)) + zC;

        // Clamp to int8_t range
        if (rounded > 127) rounded = 127;
        else if (rounded < -128) rounded = -128;

        C[row * N + col] = static_cast<int8_t>(rounded);
    }
}

extern "C" void solve(const int8_t* h_A, const int8_t* h_B, int8_t* h_C,
                      int M, int N, int K,
                      float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C) {
    int8_t *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(int8_t);
    size_t sizeB = K * N * sizeof(int8_t);
    size_t sizeC = M * N * sizeof(int8_t);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    quantized_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C,
                                                 M, N, K,
                                                 scale_A, scale_B, scale_C,
                                                 zero_point_A, zero_point_B, zero_point_C);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
