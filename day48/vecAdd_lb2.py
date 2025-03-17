#!POPCORN leaderboard vectoradd

import torch
import torch.utils.cpp_extension as cpp_extension

# CUDA Kernel: Optimized Vector Addition with FP16 Intrinsics
vector_add_cuda = cpp_extension.load_inline(
    name="vector_add_cuda",
    sources=[
        """
        extern "C" __global__ void vector_add_kernel(
            const half* A, const half* B, half* C, int N) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                C[idx] = __hadd(A[idx], B[idx]);  // FP16 addition using CUDA intrinsic
            }
        }
        """
    ],
    extra_cuda_cflags=["--use_fast_math", "-O2"],
    verbose=True,
)

def custom_kernel(data):
    """
    Optimized CUDA-based float16 vector addition.

    Args:
        data (tuple[torch.Tensor, torch.Tensor]): Two float16 tensors of shape (N, N)
    
    Returns:
        torch.Tensor: A float16 tensor of shape (N, N) containing element-wise sum.
    """
    A, B = data  # Unpack input tensors
    N = A.numel()  # Total number of elements

    # Allocate output tensor
    C = torch.empty_like(A, dtype=torch.float16, device="cuda")

    # Define block and grid size
    BLOCK_SIZE = 256
    GRID_SIZE = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch CUDA kernel
    vector_add_cuda.vector_add_kernel(
        A.contiguous(), B.contiguous(), C, N,
        grid=(GRID_SIZE,), block=(BLOCK_SIZE,)
    )

    return C
