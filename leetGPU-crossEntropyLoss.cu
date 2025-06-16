// cross_entropy_loss.cu
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

__global__ void cross_entropy_kernel(const float* logits, const int* labels, float* losses, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const float* sample_logits = logits + idx * C;

        float max_logit = sample_logits[0];
        for (int j = 1; j < C; ++j)
            if (sample_logits[j] > max_logit)
                max_logit = sample_logits[j];

        float sum_exp = 0.0f;
        for (int j = 0; j < C; ++j)
            sum_exp += expf(sample_logits[j] - max_logit);

        float log_prob = sample_logits[labels[idx]] - max_logit - logf(sum_exp);
        losses[idx] = -log_prob;
    }
}

__global__ void reduce_mean(const float* losses, float* result, int N) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    shared[tid] = (i < N) ? losses[i] : 0.0f;
    __syncthreads();

    // reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, shared[0]);
}

extern "C" void solve(const float* h_logits, const int* h_labels, int N, int C, float* h_loss) {
    float *d_logits, *d_losses, *d_loss;
    int* d_labels;

    cudaMalloc(&d_logits, N * C * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_losses, N * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    cudaMemcpy(d_logits, h_logits, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cross_entropy_kernel<<<blocks, threads>>>(d_logits, d_labels, d_losses, N, C);

    reduce_mean<<<blocks, threads>>>(d_losses, d_loss, N);

    float total_loss;
    cudaMemcpy(&total_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    *h_loss = total_loss / N;

    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_losses);
    cudaFree(d_loss);
}
