// logistic_regression.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define MAX_ITERS 1000
#define LR 0.01f

__device__ float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// Compute sigmoid(X * beta) for each sample
__global__ void compute_predictions(const float* X, const float* beta, float* preds, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = 0.0f;
        for (int j = 0; j < d; ++j)
            z += X[i * d + j] * beta[j];
        preds[i] = sigmoid(z);
    }
}

// Compute gradient = Xᵗ (y - preds)
__global__ void compute_gradient(const float* X, const float* y, const float* preds,
                                 float* grad, int n, int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += X[i * d + j] * (y[i] - preds[i]);
        grad[j] = sum;
    }
}

// Update beta += lr * grad
__global__ void update_weights(float* beta, const float* grad, int d, float lr) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) {
        beta[j] += lr * grad[j];
    }
}

extern "C" void solve(const float* h_X, const float* h_y, float* h_beta,
                      int n_samples, int n_features) {
    float *d_X, *d_y, *d_beta, *d_preds, *d_grad;
    size_t X_sz = n_samples * n_features * sizeof(float);
    size_t y_sz = n_samples * sizeof(float);
    size_t beta_sz = n_features * sizeof(float);

    cudaMalloc(&d_X, X_sz);
    cudaMalloc(&d_y, y_sz);
    cudaMalloc(&d_beta, beta_sz);
    cudaMalloc(&d_preds, y_sz);
    cudaMalloc(&d_grad, beta_sz);

    cudaMemcpy(d_X, h_X, X_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, y_sz, cudaMemcpyHostToDevice);
    cudaMemset(d_beta, 0, beta_sz);  // Initialize beta to zero

    dim3 threads(256);
    dim3 blocks_preds((n_samples + 255) / 256);
    dim3 blocks_feats((n_features + 255) / 256);

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        compute_predictions<<<blocks_preds, threads>>>(d_X, d_beta, d_preds, n_samples, n_features);
        compute_gradient<<<blocks_feats, threads>>>(d_X, d_y, d_preds, d_grad, n_samples, n_features);
        update_weights<<<blocks_feats, threads>>>(d_beta, d_grad, n_features, LR);
    }

    cudaMemcpy(h_beta, d_beta, beta_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_beta);
    cudaFree(d_preds);
    cudaFree(d_grad);
}
