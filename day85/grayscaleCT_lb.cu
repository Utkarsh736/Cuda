//!POPCORN leaderboard grayscale

#include <cuda_runtime.h>

// CUDA kernel: convert RGB image to grayscale
__global__ void grayscale_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int width, int height) {
    // 2D thread indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;  
    // Bounds check
    if (x < width && y < height) {  
        int idx = (y * width + x) * 3;  
        // Apply standard coefficients
        float r = input[idx];       // red channel  
        float g = input[idx + 1];   // green channel  
        float b = input[idx + 2];   // blue channel  
        output[y * width + x] = 
            0.2989f * r +            // R coefficient 0  
            0.5870f * g +            // G coefficient 1  
            0.1140f * b;             // B coefficient 2  
    }
}

// Host wrapper invoked by the harness
void custom_kernel(const torch::Tensor& input_tensor,
                   torch::Tensor& output_tensor) {
    // Ensure CUDA tensors
    const float* input  = input_tensor.data_ptr<float>();  
    float*       output = output_tensor.data_ptr<float>();  
    int width  = input_tensor.size(1);  
    int height = input_tensor.size(0);  

    // Define 16×16 thread blocks for good occupancy 3
    dim3 block(16, 16);  
    // Compute grid to cover entire image  
    dim3 grid((width + block.x - 1) / block.x,  
              (height + block.y - 1) / block.y);  

    // Launch kernel  
    grayscale_kernel<<<grid, block>>>(input, output, width, height);  
    // Synchronize to ensure completion 4
    cudaDeviceSynchronize();  
}