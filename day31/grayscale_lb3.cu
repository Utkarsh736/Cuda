//!POPCORN leaderboard grayscale
#include <cuda_runtime.h>
#include "task.h"
#include "utils.h"

#define BLOCK_SIZE 16  // 16x16 thread blocks for 2D grid

// CUDA kernel for RGB to Grayscale conversion
__global__ void rgb_to_grayscale_kernel(const float* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;  // Index for RGB channels
        gray[y * width + x] = (0.2989 * rgb[idx]) + (0.5870 * rgb[idx + 1]) + (0.1140 * rgb[idx + 2]);
    }
}

// Wrapper function to call CUDA kernel
output_t custom_kernel(input_t data) {
    int height = data.size(0);
    int width = data.size(1);
    int img_size = height * width;

    // Allocate memory on device
    float* d_rgb, * d_gray;
    cudaMalloc((void**)&d_rgb, img_size * 3 * sizeof(float));
    cudaMalloc((void**)&d_gray, img_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_rgb, data.data_ptr<float>(), img_size * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block size
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    rgb_to_grayscale_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize();

    // Copy results back to host
    torch::Tensor output = torch::empty({height, width}, torch::kFloat32);
    cudaMemcpy(output.data_ptr<float>(), d_gray, img_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rgb);
    cudaFree(d_gray);

    return output;
}
