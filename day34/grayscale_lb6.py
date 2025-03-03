#!POPCORN leaderboard grayscale

from task import input_t, output_t
import torch
import torch.cuda as cuda

# CUDA Kernel for grayscale conversion
grayscale_kernel = cuda.compile_raw_kernel('''
extern "C" __global__ void grayscale_kernel(const float* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        gray[y * width + x] = 0.2989f * rgb[idx] + 0.5870f * rgb[idx + 1] + 0.1140f * rgb[idx + 2];
    }
}
''', 'grayscale_kernel')

def custom_kernel(data: input_t) -> output_t:
    """
    Optimized CUDA-based RGB to Grayscale conversion.
    
    Args:
        data (input_t): RGB tensor of shape (H, W, 3) with values in [0, 1]
        
    Returns:
        output_t: Grayscale tensor of shape (H, W) with values in [0, 1]
    """
    H, W, _ = data.shape
    gray = torch.empty((H, W), device=data.device, dtype=torch.float32)
    
    # Define CUDA grid and block sizes
    BLOCK_SIZE = 16
    grid = ((W + BLOCK_SIZE - 1) // BLOCK_SIZE, (H + BLOCK_SIZE - 1) // BLOCK_SIZE)
    block = (BLOCK_SIZE, BLOCK_SIZE)
    
    # Launch CUDA kernel
    grayscale_kernel(block, grid, (data.contiguous().data_ptr(), gray.data_ptr(), W, H))
    
    return gray
