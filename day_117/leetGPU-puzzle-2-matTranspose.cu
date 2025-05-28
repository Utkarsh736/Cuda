#include <cuda_runtime.h>
#include <iostream>

__global__ void transposeKernel(const float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (x < cols && y < rows) {
        int inputIdx = y * cols + x;
        int outputIdx = x * rows + y;
        output[outputIdx] = input[inputIdx];
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
