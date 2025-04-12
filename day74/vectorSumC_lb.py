import torch
import triton
import triton.language as tl

@triton.jit
def vectorsum_kernel(
    # Pointers to input and output
    x_ptr,
    output_ptr,
    # Size of the input vector
    n_elements,
    # Block size for reduction
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing the sum of all elements in a vector.
    
    Arguments:
        x_ptr: Pointer to the input tensor
        output_ptr: Pointer to the output (scalar)
        n_elements: Number of elements in the input tensor
        BLOCK_SIZE: Number of elements to process per block
    """
    # Program ID (block index)
    pid = tl.program_id(axis=0)
    
    # Compute the block start offset
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the elements this block will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where the block extends beyond the input size
    mask = offsets < n_elements
    
    # Load the elements for this block
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the sum for this block
    block_sum = tl.sum(x)
    
    # Use atomic add to accumulate the block sum into the output
    tl.atomic_add(output_ptr, block_sum)

def vectorsum(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of all elements in the input tensor using Triton.
    
    Arguments:
        x: Input tensor of shape (N,)
        
    Returns:
        A scalar tensor containing the sum of all elements
    """
    # Ensure input is a 1D tensor
    assert x.dim() == 1, f"Input tensor must be 1-dimensional, but got shape {x.shape}"
    
    # Get the number of elements
    n_elements = x.numel()
    
    # Define block size for the kernel
    BLOCK_SIZE = 1024
    
    # Create output tensor to store the result
    output = torch.zeros(1, dtype=x.dtype, device=x.device)
    
    # Calculate grid size based on the number of elements and block size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    vectorsum_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return the scalar result
    return output.item()

# Alternative implementation with multiple-level reduction for better performance
@triton.jit
def vectorsum_kernel_two_level(
    # Pointers to input and output
    x_ptr,
    temp_ptr,
    # Size of the input vector
    n_elements,
    # Block size for reduction
    BLOCK_SIZE: tl.constexpr,
):
    """
    First level kernel for computing partial sums.
    
    Arguments:
        x_ptr: Pointer to the input tensor
        temp_ptr: Pointer to temporary storage for partial sums
        n_elements: Number of elements in the input tensor
        BLOCK_SIZE: Number of elements to process per block
    """
    # Program ID (block index)
    pid = tl.program_id(axis=0)
    
    # Compute the block start offset
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the elements this block will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where the block extends beyond the input size
    mask = offsets < n_elements
    
    # Load the elements for this block
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the sum for this block
    block_sum = tl.sum(x)
    
    # Store the block sum to temporary storage
    tl.store(temp_ptr + pid, block_sum)

@triton.jit
def final_reduction_kernel(
    temp_ptr,
    output_ptr,
    n_blocks,
):
    """
    Second level kernel to sum up all partial sums.
    
    Arguments:
        temp_ptr: Pointer to temporary storage with partial sums
        output_ptr: Pointer to the output (scalar)
        n_blocks: Number of blocks (partial sums)
    """
    # Load all partial sums - assuming this fits in a single block
    partial_sums = tl.load(temp_ptr + tl.arange(0, n_blocks))
    
    # Compute final sum
    final_sum = tl.sum(partial_sums)
    
    # Store the result to output
    tl.store(output_ptr, final_sum)

def vectorsum_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of all elements in the input tensor using an optimized two-level reduction.
    
    Arguments:
        x: Input tensor of shape (N,)
        
    Returns:
        A scalar value equal to the sum of all elements
    """
    # Ensure input is a 1D tensor
    assert x.dim() == 1, f"Input tensor must be 1-dimensional, but got shape {x.shape}"
    
    # Get the number of elements
    n_elements = x.numel()
    
    # Define block size for the kernel
    BLOCK_SIZE = 1024
    
    # Calculate the number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary storage for partial sums
    temp_storage = torch.zeros(n_blocks, dtype=x.dtype, device=x.device)
    
    # Create output tensor to store the final result
    output = torch.zeros(1, dtype=x.dtype, device=x.device)
    
    # Launch the first kernel to compute partial sums
    vectorsum_kernel_two_level[(n_blocks,)](
        x_ptr=x,
        temp_ptr=temp_storage,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch the second kernel to compute the final sum
    if n_blocks > 1024:
        # For very large inputs, we might need a multi-level reduction
        # This implementation handles the common case where n_blocks <= 1024
        raise NotImplementedError("Input size too large for this implementation")
    
    final_reduction_kernel[(1,)](
        temp_ptr=temp_storage,
        output_ptr=output,
        n_blocks=n_blocks,
    )
    
    # Return the scalar result
    return output.item()

# Example usage to test the implementation
def test_vectorsum():
    # Set vector size
    N = 1_000_000
    
    # Create input tensor with normal distribution (mean=0, std=1)
    x = torch.randn(N, device='cuda')
    
    # Compute reference result using PyTorch
    ref_output = x.sum().item()
    
    # Compute result using our Triton kernel
    triton_output = vectorsum(x)
    
    # Compute result using optimized Triton kernel
    triton_optimized_output = vectorsum_optimized(x)
    
    # Verify correctness
    assert abs(ref_output - triton_output) < 1e-5, f"Results don't match! PyTorch: {ref_output}, Triton: {triton_output}"
    assert abs(ref_output - triton_optimized_output) < 1e-5, f"Results don't match! PyTorch: {ref_output}, Triton optimized: {triton_optimized_output}"
    
    print("✓ Test passed!")
    print(f"PyTorch sum: {ref_output}")
    print(f"Triton sum: {triton_output}")
    print(f"Triton optimized sum: {triton_optimized_output}")

if __name__ == "__main__":
    test_vectorsum()