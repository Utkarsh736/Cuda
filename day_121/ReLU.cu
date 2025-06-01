#include <cuda_runtime.h>
#include <iostream>

__global__ void reluKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

void solve(const float* input, float* output, int N) {
    float *d_input, *d_output;
    size_t size = N * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
