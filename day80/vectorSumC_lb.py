#!POPCORN leaderboard vectorsum

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def sum_reduce_kernel(
    x_ptr,           # Pointer to input vector
    out_ptr,         # Pointer to output scalar
    n_elements,      # Number of elements in x
    BLOCK_SIZE: tl.constexpr  # Block size (must be power of two)
):
    # Identify which block this is
    pid = tl.program_id(0)  # 1D grid over blocks

    # Compute the start and offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask out‐of‐bounds

    # Load elements into registers
    x = tl.load(x_ptr + offsets, mask=mask)

    # Perform tree‐based reduction in registers
    # First, pairwise sum with stride=BLOCK_SIZE//2, then //4, etc.
    # This works only within a block of size <=1024 typically.
    size = BLOCK_SIZE
    while size > 1:
        half = size // 2
        # Gather partner elements
        y = tl.load(x_ptr + offsets + half, mask=(offsets + half < n_elements))
        # Sum pairwise
        x = x + y
        size = half
        tl.barrier()  # Synchronize threads within block

    # Thread 0 of each block atomically adds its partial sum
    if tl.where(pid < tl.program_id(0), True, False):  # ensure thread 0 only
        tl.atomic_add(out_ptr, 0, x[0], mask=True)

def custom_kernel(data: input_t) -> output_t:
    """
    Triton vector sum reduction. Returns a scalar sum of data.
    """
    x = data.contiguous().cuda()
    n = x.numel()
    # Create output tensor (initialized to zero)
    out = torch.zeros(1, device=x.device, dtype=x.dtype)

    # Launch one program per BLOCK
    BLOCK_SIZE = 1024
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Pointers
    x_ptr = x.data_ptr()
    out_ptr = out.data_ptr()

    # Launch the kernel
    sum_reduce_kernel[grid](
        x_ptr, out_ptr, n, BLOCK_SIZE=BLOCK_SIZE
    )
    return out