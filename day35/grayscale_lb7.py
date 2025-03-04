#!POPCORN leaderboard grayscale

import torch
import cupy as cp

# Define the CUDA kernel using CuPy
kernel_code = '''
extern "C" __global__ void grayscale_kernel(const float* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        gray[y * width + x] = 0.2989f * rgb[idx] + 0.5870f * rgb[idx + 1] + 0.1140f * rgb[idx + 2];
    }
}
'''
grayscale_kernel = cp.RawKernel(kernel_code, 'grayscale_kernel')

def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    """
    Optimized CUDA-based RGB to Grayscale conversion using CuPy.
    
    Args:
        data (torch.Tensor): RGB tensor of shape (H, W, 3) with values in [0, 1]
        
    Returns:
        torch.Tensor: Grayscale tensor of shape (H, W) with values in [0, 1]
    """
    H, W, _ = data.shape
    gray = torch.empty((H, W), device=data.device, dtype=torch.float32)

    # Convert PyTorch tensors to CuPy arrays
    data_cp = cp.asarray(data.contiguous())
    gray_cp = cp.asarray(gray)

    # Define grid and block sizes
    BLOCK_SIZE = 16
    grid = ((W + BLOCK_SIZE - 1) // BLOCK_SIZE, (H + BLOCK_SIZE - 1) // BLOCK_SIZE)
    block = (BLOCK_SIZE, BLOCK_SIZE)

    # Launch CUDA kernel
    grayscale_kernel(grid, block, (data_cp, gray_cp, W, H))

    # Convert CuPy array back to PyTorch tensor
    return torch.as_tensor(gray_cp, device=data.device)
