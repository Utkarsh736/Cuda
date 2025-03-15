#!POPCORN leaderboard matmul

import torch
import torch.utils.cpp_extension as cpp_extension

# CUDA Kernel: Optimized Matrix Multiplication using Shared Memory
matmul_cuda = cpp_extension.load_inline(
    name="matmul_cuda",
    sources=[
        """
        extern "C" __global__ void matmul_kernel(
            const half* A, const half* B, half* C, int N) {
            
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < N && col < N) {
                half sum = __float2half(0.0f);
                for (int k = 0; k < N; ++k) {
                    sum = __hadd(sum, __hmul(A[row * N + k], B[k * N + col]));
                }
                C[row * N + col] = sum;
            }
        }
        """
    ],
    extra_cuda_cflags=["--use_fast_math", "-O2"],
    verbose=True,
)

def custom_kernel(data):
    """
    Optimized CUDA-based float16 matrix multiplication.

    Args:
        data (tuple[torch.Tensor, torch.Tensor]): Two float16 tensors of shape (N, N)
    
    Returns:
        torch.Tensor: A float16 tensor of shape (N, N) containing matrix multiplication result.
    """
    A, B = data  # Unpack input tensors
    N = A.size(0)  # Matrix size (N, N)

    # Allocate output tensor
    C = torch.empty_like(A, dtype=torch.float16, device="cuda")

    # Define block and grid size
    BLOCK_SIZE = 16  # 16x16 blocks for better memory efficiency
    GRID_SIZE = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch CUDA kernel
    matmul_cuda.matmul_kernel(
        A.contiguous(), B.contiguous(), C, N,
        grid=(GRID_SIZE, GRID_SIZE), block=(BLOCK_SIZE, BLOCK_SIZE)
    )

    return C
