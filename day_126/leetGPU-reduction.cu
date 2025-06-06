#include <cuda_runtime.h>
#include <iostream>

__global__ void reduceKernel(const float* g_input, float* g_output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + tid;

    float sum = 0.0f;
    if (idx < N) sum = g_input[idx];
    if (idx + blockDim.x < N) sum += g_input[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Sequential addressing reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // First thread writes result
    if (tid == 0) {
        g_output[blockIdx.x] = sdata[0];
    }
}

void solve(const float* input, int N, float* output) {
    // One block-phase sum, then final reduction on host
    const int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);

    float *d_input, *d_intermediate;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_intermediate, blocks * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    size_t smem = threads * sizeof(float);
    reduceKernel<<<blocks, threads, smem>>>(d_input, d_intermediate, N);

    float* h_inter = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_inter, d_intermediate, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        sum += h_inter[i];
    }

    *output = sum;

    free(h_inter);
    cudaFree(d_input);
    cudaFree(d_intermediate);
}
