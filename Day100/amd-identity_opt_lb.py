#!POPCORN leaderboard amd-identity

import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Number of scalar elements per thread (vector width)
VEC = 4
# Number of threads per block
BLOCK = 256

@triton.jit
def identity_kernel(in_ptr, out_ptr, size, stride, **meta):
    """
    in_ptr, out_ptr: pointers to input/output float tensors
    size: total number of elements
    stride: stride between consecutive elements (for flat tensors it's 1)
    BLOCK and VEC are constexpr meta-parameters
    """
    pid = tl.program_id(0)
    # Each block handles BLOCK * VEC elements
    base = pid * meta['BLOCK'] * meta['VEC']
    offs = base + tl.arange(0, meta['BLOCK'] * meta['VEC'])
    mask = offs < size

    # Load VEC-wide vectors as float
    x = tl.load(in_ptr + offs * stride, mask=mask)
    tl.store(out_ptr + offs * stride, x, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    inp, out = data
    # Ensure contiguous
    inp = inp.contiguous()
    out = out.contiguous()
    # Flatten
    size = inp.numel()
    # Get pointers
    in_ptr  = inp.data_ptr()
    out_ptr = out.data_ptr()

    # Compute grid size
    grid = (triton.cdiv(size, BLOCK * VEC),)

    # Launch
    identity_kernel[grid](
        in_ptr, out_ptr,
        size, 1,
        BLOCK=BLOCK, VEC=VEC
    )
    return out