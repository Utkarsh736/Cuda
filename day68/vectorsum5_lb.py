import torch
import triton
import triton.language as tl

@triton.jit
def vector_sum_kernel(
    x_ptr,        # pointer to input vector
    output_ptr,   # pointer to output scalar
    n_elements,   # number of elements in the vector
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute the sum of all elements in a vector using Triton's block-level reduction.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Compute block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where the block extends beyond the vector
    mask = offsets < n_elements
    
    # Load elements from the input vector where mask is valid
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform parallel reduction within the block
    block_sum = tl.sum(x, axis=0)
    
    # Use atomic add to accumulate the results from all blocks
    tl.atomic_add(output_ptr, block_sum)

def vector_sum(x):
    """
    Compute the sum of all elements in a vector using Triton.
    
    Args:
        x: PyTorch tensor of shape (N,)
        
    Returns:
        PyTorch scalar representing the sum of all elements
    """
    # Input validation
    assert x.dim() == 1, "Input must be a 1D tensor"
    
    # Ensure input is on GPU and contiguous
    x = x.contiguous()
    
    # Create output tensor (scalar) initialized to zero
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    
    # Configure kernel
    BLOCK_SIZE = 1024  # Process 1024 elements per thread block
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    vector_sum_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return scalar value
    return output.item()