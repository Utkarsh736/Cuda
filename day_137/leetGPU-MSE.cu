// mse_loss.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void mse_kernel(const float* preds, const float* targets, float* partial_sums, int N) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float temp = 0.0f;

    if (i < N) {
        float diff = preds[i] - targets[i];
        temp = diff * diff;
    }

    cache[tid] = temp;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = cache[0];
}

extern "C" void solve(const float* h_predictions, const float* h_targets, int N, float* h_mse) {
    float *d_predictions, *d_targets, *d_partial, *h_partial;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaMalloc(&d_predictions, N * sizeof(float));
    cudaMalloc(&d_targets, N * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    h_partial = (float*)malloc(blocks * sizeof(float));

    cudaMemcpy(d_predictions, h_predictions, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, N * sizeof(float), cudaMemcpyHostToDevice);

    mse_kernel<<<blocks, threads>>>(d_predictions, d_targets, d_partial, N);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; ++i)
        total += h_partial[i];

    *h_mse = total / N;

    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_partial);
    free(h_partial);
}
