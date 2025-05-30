#include <cuda_runtime.h>
#include <iostream>

__global__ void conv1dKernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (idx < output_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            sum += input[idx + k] * kernel[kernel_size - 1 - k]; // Flip kernel for convolution
        }
        output[idx] = sum;
    }
}

void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    size_t input_bytes = input_size * sizeof(float);
    size_t kernel_bytes = kernel_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, input_bytes);
    cudaMalloc((void**)&d_kernel, kernel_bytes);
    cudaMalloc((void**)&d_output, output_bytes);

    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    conv1dKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_size, kernel_size);

    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
