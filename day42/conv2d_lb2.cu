//!POPCORN leaderboard conv2d

#include <cuda_runtime.h>
#include "task.h"
#include "utils.h"

#define BLOCK_SIZE 16  // 16x16 thread block for efficient shared memory usage

// CUDA Kernel for 2D Convolution
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                              int batch, int channels, int height, int width, 
                              int kernel_size, int output_height, int output_width) {
    // Calculate global row and column indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Loop over batches and channels
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            if (x < output_width && y < output_height) {
                float sum = 0.0f;

                // Perform convolution
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_x = x + kx;
                        int in_y = y + ky;
                        int input_idx = ((b * channels + c) * height + in_y) * width + in_x;
                        int kernel_idx = ((c * channels + c) * kernel_size + ky) * kernel_size + kx;
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }

                // Store result
                int output_idx = ((b * channels + c) * output_height + y) * output_width + x;
                output[output_idx] = sum;
            }
        }
    }
}

// Wrapper function to call CUDA kernel
output_t custom_kernel(input_t data) {
    auto input_tensor = data.first;
    auto kernel_tensor = data.second;

    int batch = input_tensor.size(0);
    int channels = input_tensor.size(1);
    int height = input_tensor.size(2);
    int width = input_tensor.size(3);
    int kernel_size = kernel_tensor.size(2);

    int output_height = height - kernel_size + 1;
    int output_width = width - kernel_size + 1;

    // Allocate memory on device
    float* d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, batch * channels * height * width * sizeof(float));
    cudaMalloc((void**)&d_kernel, channels * channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, batch * channels * output_height * output_width * sizeof(float));

    // Copy input and kernel data to device
    cudaMemcpy(d_input, input_tensor.data_ptr<float>(), batch * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel_tensor.data_ptr<float>(), channels * channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    conv2d_kernel<<<grid, block>>>(d_input, d_kernel, d_output, batch, channels, height, width, kernel_size, output_height, output_width);
    cudaDeviceSynchronize();

    // Copy results back to host
    torch::Tensor output = torch::empty({batch, channels, output_height, output_width}, torch::kFloat32);
    cudaMemcpy(output.data_ptr<float>(), d_output, batch * channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return output;
}
