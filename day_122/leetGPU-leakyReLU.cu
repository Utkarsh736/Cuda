#include <cuda_runtime.h>
#include <iostream>

__global__ void leakyReluKernel(const float* input, float* output, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = (x >= 0.0f) ? x : alpha * x;
    }
}

void solve(const float* input, float* output, int N) {
    const float alpha = 0.01f;
    float *d_input, *d_output;
    size_t size = N * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leakyReluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, alpha);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
