import torch
import triton
import triton.language as tl

@triton.jit
def prefix_sum_kernel(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute local prefix sum within block
    sum = x
    for i in range(1, BLOCK_SIZE):
        prev_sum = tl.where(offsets < block_start + i, sum, 0.0)
        sum = prev_sum + x if i == 1 else sum + tl.where(offsets >= block_start + i, x, 0.0)
    
    # Store block sums
    block_sums = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if pid > 0:
        block_sums = tl.load(output_ptr + block_start - 1, mask=mask, other=0.0)
    
    # Add block sums to get final prefix sum
    result = sum + block_sums
    tl.store(output_ptr + offsets, result, mask=mask)

def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    n = data.shape[0]
    output = torch.empty_like(data)
    
    # Configure grid
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    prefix_sum_kernel[grid](
        input_ptr=data,
        output_ptr=output,
        n=n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
