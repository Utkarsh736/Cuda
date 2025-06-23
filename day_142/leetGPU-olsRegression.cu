// ols_regression.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Matrix transpose kernel: B = Aᵀ
__global__ void transpose(const float* A, float* B, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols)
        B[j * rows + i] = A[i * cols + j];
}

// Matrix multiplication: C = A × B (row-major)
__global__ void matmul(const float* A, const float* B, float* C,
                       int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

// Naive matrix inverse using Gauss-Jordan (for small feature sizes only)
__global__ void invert(float* A, float* Ainv, int N) {
    int tid = threadIdx.x;
    for (int i = 0; i < N; ++i) {
        float diag = A[i * N + i];
        for (int j = 0; j < N; ++j) {
            A[i * N + j] /= diag;
            Ainv[i * N + j] /= diag;
        }
        __syncthreads();
        for (int k = 0; k < N; ++k) {
            if (k == i) continue;
            float factor = A[k * N + i];
            for (int j = 0; j < N; ++j) {
                A[k * N + j] -= factor * A[i * N + j];
                Ainv[k * N + j] -= factor * Ainv[i * N + j];
            }
        }
        __syncthreads();
    }
}

extern "C" void solve(const float* h_X, const float* h_y, float* h_beta,
                      int n_samples, int n_features) {
    float *d_X, *d_XT, *d_XTX, *d_XTy, *d_XTX_inv, *d_beta, *d_y;

    size_t X_sz = n_samples * n_features * sizeof(float);
    size_t XT_sz = n_features * n_samples * sizeof(float);
    size_t XTX_sz = n_features * n_features * sizeof(float);
    size_t XTy_sz = n_features * sizeof(float);
    size_t y_sz = n_samples * sizeof(float);
    size_t beta_sz = n_features * sizeof(float);

    cudaMalloc(&d_X, X_sz);
    cudaMalloc(&d_XT, XT_sz);
    cudaMalloc(&d_XTX, XTX_sz);
    cudaMalloc(&d_XTy, XTy_sz);
    cudaMalloc(&d_XTX_inv, XTX_sz);
    cudaMalloc(&d_y, y_sz);
    cudaMalloc(&d_beta, beta_sz);

    cudaMemcpy(d_X, h_X, X_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, y_sz, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks1((n_samples + 15) / 16, (n_features + 15) / 16);
    transpose<<<blocks1, threads>>>(d_X, d_XT, n_samples, n_features);

    dim3 blocks2((n_features + 15) / 16, (n_features + 15) / 16);
    matmul<<<blocks2, threads>>>(d_XT, d_X, d_XTX, n_features, n_samples, n_features);

    // Invert XTX
    float* temp_XTX = (float*)malloc(XTX_sz);
    float* temp_I = (float*)malloc(XTX_sz);
    cudaMemcpy(temp_XTX, d_XTX, XTX_sz, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_features * n_features; ++i) temp_I[i] = 0;
    for (int i = 0; i < n_features; ++i) temp_I[i * n_features + i] = 1.0f;

    cudaMemcpy(d_XTX, temp_XTX, XTX_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_XTX_inv, temp_I, XTX_sz, cudaMemcpyHostToDevice);
    invert<<<1, 1>>>(d_XTX, d_XTX_inv, n_features);

    dim3 blocks3((1 + 15) / 16, (n_features + 15) / 16);
    matmul<<<blocks3, threads>>>(d_XT, d_y, d_XTy, n_features, n_samples, 1);
    matmul<<<blocks3, threads>>>(d_XTX_inv, d_XTy, d_beta, n_features, n_features, 1);

    cudaMemcpy(h_beta, d_beta, beta_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_X); cudaFree(d_XT); cudaFree(d_XTX); cudaFree(d_XTX_inv);
    cudaFree(d_XTy); cudaFree(d_y); cudaFree(d_beta);
    free(temp_XTX); free(temp_I);
}
