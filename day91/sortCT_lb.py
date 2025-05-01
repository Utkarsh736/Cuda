#!POPCORN leaderboard sort

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def bitonic_sort_kernel(x_ptr, n, stage: tl.constexpr, substage: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid

    partner = idx ^ (1 << substage)
    if idx < n and partner < n:
        x_i = tl.load(x_ptr + idx)
        x_p = tl.load(x_ptr + partner)

        up = ((idx >> stage) & 1) == 0
        cond = (x_i > x_p) if up else (x_i < x_p)

        new_i = tl.where(cond, x_p, x_i)
        new_p = tl.where(cond, x_i, x_p)

        tl.store(x_ptr + idx, new_i)
        tl.store(x_ptr + partner, new_p)

def custom_kernel(data: input_t) -> output_t:
    x = data.contiguous().cuda()
    n = x.numel()
    m = 1 << (n - 1).bit_length()

    if m != n:
        pad = torch.full((m - n,), float('inf'), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad])

    for stage in range(1, m.bit_length()):
        for sub in range(stage):
            bitonic_sort_kernel[(m,)](
                x.data_ptr(), m, stage, sub
            )

    return x[:n]