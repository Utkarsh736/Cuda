#include <cuda_runtime.h>
#include <iostream>

__global__ void reverseKernel(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int mirror_idx = N - 1 - idx;

    if (idx < N / 2) {
        float temp = input[idx];
        input[idx] = input[mirror_idx];
        input[mirror_idx] = temp;
    }
}

void solve(float* input, int N) {
    float* d_input;
    size_t size = N * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    reverseKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);

    cudaMemcpy(input, d_input, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
}
