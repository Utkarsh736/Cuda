#!POPCORN leaderboard grayscale

import torch
import torch.utils.cpp_extension as cpp_extension
import os

# Define the CUDA kernel as a string
cuda_source = """
#include <torch/extension.h>

extern "C" __global__ void grayscale_kernel(
    const float* __restrict__ rgb, float* __restrict__ gray, 
    int width, int height) {

    // Calculate the global thread positions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we don't access out-of-bounds memory
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;  // Index for RGB channels
        gray[y * width + x] = 0.2989f * rgb[idx] + 
                              0.5870f * rgb[idx + 1] + 
                              0.1140f * rgb[idx + 2];
    }
}

// C++ wrapper function to be called from Python
torch::Tensor grayscale_cuda_forward(torch::Tensor rgb) {
    const auto height = rgb.size(0);
    const auto width = rgb.size(1);
    
    // Create output tensor on the same device
    auto gray = torch::empty({height, width}, 
                            rgb.options().dtype(torch::kFloat32));
    
    // Calculate grid and block dimensions
    const int BLOCK_SIZE = 16;
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((width + threads.x - 1) / threads.x,
                     (height + threads.y - 1) / threads.y);
    
    // Launch kernel
    grayscale_kernel<<<blocks, threads>>>(
        rgb.data_ptr<float>(),
        gray.data_ptr<float>(),
        width, height
    );
    
    return gray;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grayscale_cuda_forward, "Grayscale forward (CUDA)");
}
"""

# Create a fallback implementation in case CUDA compilation fails
def grayscale_fallback(data):
    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

# Try to load the CUDA extension, fall back to CPU implementation if it fails
try:
    grayscale_cuda = cpp_extension.load_inline(
        name="grayscale_cuda",
        cpp_sources="",  # No separate C++ sources
        cuda_sources=cuda_source,
        functions=["forward"],
        verbose=True,
        with_cuda=True
    )
    
    HAS_CUDA = True
except Exception as e:
    print(f"Warning: Could not load CUDA extension: {e}")
    print("Falling back to CPU implementation")
    HAS_CUDA = False

def custom_kernel(data):
    """
    RGB to Grayscale conversion using CUDA if available.
    
    Args:
        data (torch.Tensor): RGB tensor of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        torch.Tensor: Grayscale tensor of shape (H, W) with values in [0, 1]
    """
    if HAS_CUDA and torch.cuda.is_available() and data.is_cuda:
        # Use CUDA implementation
        return grayscale_cuda.forward(data)
    else:
        # Use fallback implementation
        return grayscale_fallback(data)
