// multihead_attention.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__device__ float dot_product(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) sum += a[i] * b[i];
    return sum;
}

__device__ void softmax(float* scores, int N) {
    float max_val = -1e9f;
    for (int i = 0; i < N; ++i)
        if (scores[i] > max_val) max_val = scores[i];

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }
    for (int i = 0; i < N; ++i)
        scores[i] /= sum;
}

__global__ void mha_kernel(const float* Q, const float* K, const float* V,
                           float* output, int N, int d_model, int h) {
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= N) return;

    int d_head = d_model / h;
    extern __shared__ float shared_mem[];  // softmax scores

    for (int head = 0; head < h; ++head) {
        const float* q = Q + token_id * d_model + head * d_head;

        float* scores = shared_mem;  // N per head

        // Compute scores[token_id][i] = q · k_i / sqrt(d_head)
        for (int i = 0; i < N; ++i) {
            const float* k = K + i * d_model + head * d_head;
            scores[i] = dot_product(q, k, d_head) / sqrtf((float)d_head);
        }

        softmax(scores, N);

        // attention = sum_i softmax[i] * v_i
        float* out = output + token_id * d_model + head * d_head;
        for (int j = 0; j < d_head; ++j) {
            float val = 0.0f;
            for (int i = 0; i < N; ++i) {
                const float* v = V + i * d_model + head * d_head;
                val += scores[i] * v[j];
            }
            out[j] = val;
        }
        __syncthreads();
    }
}

extern "C" void solve(const float* h_Q, const float* h_K, const float* h_V,
                      float* h_output, int N, int d_model, int h) {
    float *d_Q, *d_K, *d_V, *d_output;
    size_t sz = N * d_model * sizeof(float);

    cudaMalloc(&d_Q, sz);
    cudaMalloc(&d_K, sz);
    cudaMalloc(&d_V, sz);
    cudaMalloc(&d_output, sz);

    cudaMemcpy(d_Q, h_Q, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sz, cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    size_t smem = N * sizeof(float);  // scores per thread

    mha_kernel<<<blocks, threads, smem>>>(d_Q, d_K, d_V, d_output, N, d_model, h);
    cudaMemcpy(h_output, d_output, sz, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
}
