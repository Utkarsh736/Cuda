#include <cuda_runtime.h>
#include <iostream>

__global__ void monteCarloKernel(const float* y_samples, float* partial_sums, int n_samples) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float y = (i < n_samples) ? y_samples[i] : 0.0f;
    sdata[tid] = y;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

void solve(float a, float b, const float* y_samples, int n_samples, float* result) {
    float* d_y_samples;
    float* d_partial_sums;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_samples + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    cudaMalloc((void**)&d_y_samples, n_samples * sizeof(float));
    cudaMemcpy(d_y_samples, y_samples, n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_partial_sums, blocksPerGrid * sizeof(float));

    monteCarloKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_y_samples, d_partial_sums, n_samples);

    // Copy partial sums back to host and compute final result
    float* h_partial_sums = new float[blocksPerGrid];
    cudaMemcpy(h_partial_sums, d_partial_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        sum += h_partial_sums[i];
    }

    *result = (b - a) * sum / n_samples;

    // Clean up
    delete[] h_partial_sums;
    cudaFree(d_y_samples);
    cudaFree(d_partial_sums);
}
