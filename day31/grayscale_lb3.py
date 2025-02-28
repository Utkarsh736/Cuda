#!POPCORN leaderboard grayscale

import torch
import torch.utils.cpp_extension
import os
import tempfile

# Define the CUDA code as a string
cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

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

// C++ wrapper for the CUDA kernel
torch::Tensor rgb_to_grayscale_cuda(torch::Tensor rgb) {
    // Get dimensions
    int height = rgb.size(0);
    int width = rgb.size(1);
    int img_size = height * width;
    
    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(rgb.device());
    torch::Tensor gray = torch::empty({height, width}, options);
    
    // Configure grid and block size
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    rgb_to_grayscale_kernel<<<grid, block>>>(
        rgb.data_ptr<float>(),
        gray.data_ptr<float>(),
        width,
        height
    );
    
    return gray;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rgb_to_grayscale", &rgb_to_grayscale_cuda, "RGB to Grayscale conversion (CUDA)");
}
'''

# Create a temporary directory to store the CUDA code
temp_dir = tempfile.mkdtemp()
source_file = os.path.join(temp_dir, 'rgb_to_grayscale_cuda.cu')

with open(source_file, 'w') as f:
    f.write(cuda_source)

# Load the CUDA extension
try:
    rgb_to_grayscale_module = torch.utils.cpp_extension.load(
        name="rgb_to_grayscale_cuda",
        sources=[source_file],
        verbose=True,
        is_python_module=False,
        is_cuda_extension=True
    )
    
    # The JIT compilation will create a callable function
    rgb_to_grayscale = rgb_to_grayscale_module.rgb_to_grayscale
    
except Exception as e:
    print(f"Error loading CUDA extension: {e}")
    # Fallback implementation in PyTorch
    def rgb_to_grayscale(rgb_tensor):
        return (0.2989 * rgb_tensor[:, :, 0] + 
                0.5870 * rgb_tensor[:, :, 1] + 
                0.1140 * rgb_tensor[:, :, 2])

# Main function for the leaderboard
def custom_kernel(input_tensor):
    # Ensure the input is on the correct device
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    # Apply the grayscale conversion
    if torch.cuda.is_available():
        result = rgb_to_grayscale(input_tensor)
    else:
        # Fallback to CPU implementation if CUDA is not available
        result = 0.2989 * input_tensor[:, :, 0] + 0.5870 * input_tensor[:, :, 1] + 0.1140 * input_tensor[:, :, 2]
    
    return result.to('cpu')

# Direct test function if run as a script
if __name__ == "__main__":
    # Create a sample RGB image
    sample = torch.rand(256, 256, 3, dtype=torch.float32)
    
    # Convert to grayscale
    grayscale = custom_kernel(sample)
    
    print(f"RGB shape: {sample.shape}")
    print(f"Grayscale shape: {grayscale.shape}")
