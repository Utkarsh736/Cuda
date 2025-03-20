#!POPCORN leaderboard grayscale

import torch
import torch.utils.cpp_extension as cpp_extension

# CUDA Kernel: Optimized Grayscale Conversion using Shared Memory & Memory Coalescing
grayscale_cuda = cpp_extension.load_inline(
    name="grayscale_cuda",
    sources=[
        """
        extern "C" __global__ void grayscale_kernel(
            const float* __restrict__ rgb, float* __restrict__ gray, 
            int width, int height) {

            // Shared memory for faster loads
            __shared__ float tile[48][16];  // 16x16 block + padding for memory coalescing

            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int idx = (y * width + x) * 3;

            // Load data into shared memory
            if (x < width && y < height) {
                tile[threadIdx.y * 3][threadIdx.x] = rgb[idx];
                tile[threadIdx.y * 3 + 1][threadIdx.x] = rgb[idx + 1];
                tile[threadIdx.y * 3 + 2][threadIdx.x] = rgb[idx + 2];
            }
            __syncthreads();

            // Compute grayscale value
            if (x < width && y < height) {
                gray[y * width + x] = 0.2989f * tile[threadIdx.y * 3][threadIdx.x] + 
                                      0.5870f * tile[threadIdx.y * 3 + 1][threadIdx.x] + 
                                      0.1140f * tile[threadIdx.y * 3 + 2][threadIdx.x];
            }
        }
        """
    ],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

def custom_kernel(data):
    """
    Highly Optimized CUDA-based RGB to Grayscale conversion.

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
