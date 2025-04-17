#!POPCORN leaderboard sort

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def bitonic_sort_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data into registers
    x = tl.load(x_ptr + offsets, mask=mask)

    # Bitonic sort
    for k in range(1, BLOCK_SIZE.bit_length()):
        for j in reversed(range(k)):
            stride = 1 << j
            partner = offsets ^ stride
            partner_mask = partner < n_elements
            partner_vals = tl.load(x_ptr + partner, mask=partner_mask)
            should_swap = (offsets & (1 << k)) == 0
            cmp = (x > partner_vals) if should_swap else (x < partner_vals)
            x_new = tl.where(cmp, partner_vals, x)
            x = x_new

    # Store sorted data
    tl.store(x_ptr + offsets, x, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    """
    Sorts the input 1D tensor in ascending order using a Triton kernel.

    Args:
        data (torch.Tensor): 1D tensor of floating-point numbers.

    Returns:
        torch.Tensor: Sorted 1D tensor.
    """
    x = data.contiguous().cuda()
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Must be a power of two

    # Pad input if necessary
    if n_elements % BLOCK_SIZE != 0:
        pad_size = BLOCK_SIZE - (n_elements % BLOCK_SIZE)
        x = torch.cat([x, torch.full((pad_size,), float('inf'), device=x.device, dtype=x.dtype)])
        n_elements = x.numel()

    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    bitonic_sort_kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Remove padding if added
    return x[:data.numel()]
