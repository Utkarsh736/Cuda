#include <cuda_runtime.h>
#include "task.h"
#include "utils.h"

#define BLOCK_SIZE 16  // 16x16 thread blocks

// CUDA Kernel for Tensor Parallelism + Data Parallelism
__global__ void tensor_parallel_kernel(float* input, float* output, int height, int width) {
    // Compute global thread indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int index = y * width + x;
        
        // Example operation: Scaled element-wise transformation
        output[index] = input[index] * 2.0f;
    }
}

// Wrapper function to launch CUDA kernel
output_t custom_kernel(input_t data) {
    auto input_tensor = data;
    int height = input_tensor.size(0);
    int width = input_tensor.size(1);

    // Allocate memory on device
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, height * width * sizeof(float));
    cudaMalloc((void**)&d_output, height * width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor.data_ptr<float>(), height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block sizes for parallel execution
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    tensor_parallel_kernel<<<grid, block>>>(d_input, d_output, height, width);
    cudaDeviceSynchronize();

    // Copy results back to host
    torch::Tensor output = torch::empty({height, width}, torch::kFloat32);
    cudaMemcpy(output.data_ptr<float>(), d_output, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
