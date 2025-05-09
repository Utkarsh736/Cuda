#!POPCORN leaderboard amd-identity

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def identity_kernel(in_ptr, out_ptr, size,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

def custom_kernel(data: tuple) -> torch.Tensor:
    input_tensor, output_tensor = data
    assert input_tensor.shape == output_tensor.shape
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()
    N = input_tensor.numel()

    BLOCK = 1024
    grid = lambda META: (triton.cdiv(N, META["BLOCK"]),)

    identity_kernel[grid](input_tensor, output_tensor, N, BLOCK=BLOCK)
    return output_tensor