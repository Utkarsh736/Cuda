#!POPCORN leaderboard matmul

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,               # raw pointers
    M, N, K,                           # matrix dims
    stride_am, stride_ak,              # A strides
    stride_bk, stride_bn,              # B strides
    stride_cm, stride_cn,              # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program‐level indices
    pid_m = tl.program_id(0)  # which M‐tile
    pid_n = tl.program_id(1)  # which N‐tile

    # Offsets within each tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # M rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # N cols
    offs_k = tl.arange(0, BLOCK_K)                    # K dimension

    # Initialize accumulator to zero (FP32 for precision)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Compute A block: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak
        a = tl.load(a_ptrs)                                      # 4

        # Compute B block: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs)                                      # 5

        # Accumulate dot‐product
        acc += tl.dot(a, b)                                      # 6

    # Write back result (cast to same dtype as inputs)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32))                        # 7

def custom_kernel(data: input_t) -> output_t:
    """
    Triton matmul: multiplies A @ B for A of shape (M,K) and B of shape (K,N).
    """
    A, B = data
    M, K = A.shape
    _, N = B.shape

    # Ensure CUDA, contiguous
    A = A.contiguous().cuda()                                    # 8
    B = B.contiguous().cuda()                                    # 9
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)  # 10

    # Block‐tile dimensions (multiples of 16)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32                                            

    # Launch grid
    grid = (M // BLOCK_M, N // BLOCK_N)                       # 11
    matmul_kernel[grid](
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C                                                     # 12