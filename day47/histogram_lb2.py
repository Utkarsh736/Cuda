#!POPCORN leaderboard histogram

import torch
import torch.utils.cpp_extension as cpp_extension

# CUDA Kernel: Optimized Histogram Computation
histogram_cuda = cpp_extension.load_inline(
    name="histogram_cuda",
    sources=[
        """
        extern "C" __global__ void histogram_kernel(
            const float* data, int* histogram, int size, int num_bins) {
            
            // Shared memory histogram for faster updates
            extern __shared__ int local_hist[];

            int tid = threadIdx.x + blockIdx.x * blockDim.x;

            // Initialize shared memory histogram
            for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
                local_hist[i] = 0;
            }
            __syncthreads();

            // Populate shared memory histogram
            if (tid < size) {
                int bin = int(data[tid] * num_bins / 100.0f); // Normalize input to range [0, 100]
                if (bin < num_bins) {
                    atomicAdd(&local_hist[bin], 1);
                }
            }
            __syncthreads();

            // Merge shared memory histogram into global histogram
            for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
                atomicAdd(&histogram[i], local_hist[i]);
            }
        }
        """
    ],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    """
    Optimized CUDA-based histogram computation.

    Args:
        data (torch.Tensor): A 1D tensor containing floating-point values in [0, 100]
    
    Returns:
        torch.Tensor: Histogram tensor of shape (num_bins,)
    """
    size = data.numel()
    num_bins = size // 16  # Number of bins as per leaderboard specs

    # Allocate output tensor
    hist = torch.zeros(num_bins, dtype=torch.int32, device="cuda")

    # Define block and grid size
    BLOCK_SIZE = 256
    GRID_SIZE = (size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch CUDA kernel with shared memory
    histogram_cuda.histogram_kernel(
        data.contiguous(), hist, size, num_bins,
        grid=(GRID_SIZE,), block=(BLOCK_SIZE,), shared_memory=num_bins * 4
    )

    return hist
