#!POPCORN leaderboard prefix_sum

import torch
import torch.utils.cpp_extension as cpp_extension

# CUDA Kernel: Optimized Parallel Inclusive Prefix Sum (Scan)
prefix_sum_cuda = cpp_extension.load_inline(
    name="prefix_sum_cuda",
    sources=[
        """
        extern "C" __global__ void prefix_sum_kernel(float* data, float* output, int N) {
            extern __shared__ float temp[];

            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // Load data into shared memory
            if (idx < N)
                temp[tid] = data[idx];
            else
                temp[tid] = 0;

            __syncthreads();

            // Perform inclusive scan using a reduction-like method
            for (int offset = 1; offset < blockDim.x; offset *= 2) {
                float val = 0;
                if (tid >= offset) {
                    val = temp[tid - offset];
                }
                __syncthreads();
                temp[tid] += val;
                __syncthreads();
            }

            // Store result in output array
            if (idx < N) {
                output[idx] = temp[tid];
            }
        }
        """
    ],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    """
    Optimized CUDA-based inclusive prefix sum (scan).

    Args:
        data (torch.Tensor): A 1D tensor of size N.
    
    Returns:
        torch.Tensor: A 1D tensor of size N containing the inclusive prefix sum.
    """
    N = data.numel()

    # Allocate output tensor
    output = torch.empty_like(data, device="cuda")

    # Define block and grid size
    BLOCK_SIZE = 256
    GRID_SIZE = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch CUDA kernel with shared memory
    prefix_sum_cuda.prefix_sum_kernel(
        data.contiguous(), output, N,
        grid=(GRID_SIZE,), block=(BLOCK_SIZE,), shared_memory=BLOCK_SIZE * 4
    )

    return output
