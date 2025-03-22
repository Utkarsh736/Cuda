#!POPCORN leaderboard grayscale

import torch
import torch.utils.cpp_extension as cpp_extension

# Define the CUDA kernel as a string
cuda_source = """
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
"""

# Load the CUDA extension
grayscale_cuda = cpp_extension.load_inline(
    name="grayscale_cuda",
    cpp_sources="",  # No C++ sources
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

def custom_kernel(data):
    """
    Optimized CUDA-based RGB to Grayscale conversion.

    Args:
        data (torch.Tensor): RGB tensor of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        torch.Tensor: Grayscale tensor of shape (H, W) with values in [0, 1]
    """
    H, W, _ = data.shape
    gray = torch.empty((H, W), device=data.device, dtype=torch.float32)

    # Define block and grid sizes
    BLOCK_SIZE = 16
    grid = ((W + BLOCK_SIZE - 1) // BLOCK_SIZE, (H + BLOCK_SIZE - 1) // BLOCK_SIZE)
    block = (BLOCK_SIZE, BLOCK_SIZE)

    # Launch CUDA kernel
    grayscale_cuda.grayscale_kernel(
        data.contiguous(), gray, W, H,
        grid=grid, block=block
    )

    return gray
