#include <cuda_runtime.h>
#include <torch/torch.h>

__global__ void rgb_to_grayscale_kernel(const float* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float r = rgb[idx];
        float g = rgb[idx + 1];
        float b = rgb[idx + 2];
        gray[y * width + x] = 0.2989f * r + 0.5870f * g + 0.1140f * b;
    }
}

torch::Tensor custom_kernel(torch::Tensor input) {
    // Ensure the input is on CUDA
    input = input.contiguous().to(torch::kCUDA);
    
    int height = input.size(0);
    int width = input.size(1);

    // Allocate output tensor
    auto output = torch::empty({height, width}, input.options().dtype(torch::kFloat32));

    // Get raw pointers
    const float* d_rgb = input.data_ptr<float>();
    float* d_gray = output.data_ptr<float>();

    // Define CUDA grid/block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    rgb_to_grayscale_kernel<<<numBlocks, threadsPerBlock>>>(d_rgb, d_gray, width, height);

    // Ensure the kernel execution is complete before returning
    cudaDeviceSynchronize();

    return output;
}
