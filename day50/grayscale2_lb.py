#!POPCORN leaderboard grayscale

import torch
import torch.utils.cpp_extension as cpp_extension

# CUDA Kernel: Optimized Grayscale Conversion using Element-wise Parallelism
grayscale_cuda = cpp_extension.load_inline(
    name="grayscale_cuda",
    sources=[
        """
        extern "C" __global__ void grayscale_kernel(
            const float* rgb, float* gray, int width, int height) {

            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height) {
                int idx = (y * width + x) * 3;  // RGB channels index
                gray[y * width + x] = 0.2989f * rgb[idx] + 
                                      0.5870f * rgb[idx + 1] + 
                                      0.1140f * rgb[idx + 2];
            }
        }
        """
    ],
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
