#!POPCORN leaderboard matmul

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting indices of the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks from A and B
        a_offsets = (offs_m[:, None] * stride_am) + ((k + offs_k)[None, :] * stride_ak)
        b_offsets = ((k + offs_k)[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        a = tl.load(a_ptr + a_offsets)
        b = tl.load(b_ptr + b_offsets)

        # Accumulate partial results
        acc += tl.dot(a, b)

    # Write back the result
    c_offsets = (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptr + c_offsets, acc)

def custom_kernel(data: input_t) -> output_t:
    """
    Performs matrix multiplication using a Triton kernel.

    Args:
        data (Tuple[torch.Tensor, torch.Tensor]): A tuple containing two input tensors A and B.

    Returns:
        torch.Tensor: The result of matrix multiplication A @ B.
    """
    A, B = data
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix multiplication."

    M, K = A.shape
    _, N = B.shape

    # Ensure tensors are contiguous and on CUDA
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Launch the kernel
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return C
