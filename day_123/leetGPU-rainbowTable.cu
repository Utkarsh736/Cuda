#include <cuda_runtime.h>
#include <iostream>

__device__ uint32_t simpleHash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

__global__ void hashKernel(const uint32_t* input, uint32_t* output, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t val = input[idx];
        for (int i = 0; i < R; ++i) {
            val = simpleHash(val);
        }
        output[idx] = val;
    }
}

void solve(const uint32_t* numbers, uint32_t* hashes, int N, int R) {
    uint32_t *d_input, *d_output;
    size_t size = N * sizeof(uint32_t);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, numbers, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    hashKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, R);

    cudaMemcpy(hashes, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
