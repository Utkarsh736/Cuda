import torch
import triton
import triton.language as tl

@triton.jit
def vectoradd_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N,
    # Parameters for tiling
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Kernel for performing element-wise addition of two float16 tensors.
    
    Arguments:
        a_ptr: Pointer to the first input tensor (float16)
        b_ptr: Pointer to the second input tensor (float16)
        c_ptr: Pointer to the output tensor (float16)
        M, N: Dimensions of the tensors
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes for parallelization
    """
    # Program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate the offsets for this specific block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Generate offsets for the current thread
    # Create a range from 0 to BLOCK_SIZE_N
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    # Create a range from 0 to BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    
    # Create a mesh grid using the offsets
    offs_m, offs_n = tl.meshgrid(offsets_m, offsets_n)
    
    # Compute the linear index for each element
    offs = offs_m * N + offs_n
    
    # Create masks to handle the case where the block extends beyond the tensor boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load data from tensor A
    a = tl.load(a_ptr + offs, mask=mask)
    # Load data from tensor B
    b = tl.load(b_ptr + offs, mask=mask)
    
    # Perform the element-wise addition
    c = a + b
    
    # Store the result to tensor C
    tl.store(c_ptr + offs, c, mask=mask)

def vectoradd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform element-wise addition of two float16 tensors using Triton.
    
    Arguments:
        a: First input tensor of shape (N, N) and dtype float16
        b: Second input tensor of shape (N, N) and dtype float16
        
    Returns:
        Output tensor of shape (N, N) and dtype float16
    """
    # Make sure the input tensors are of type float16
    assert a.dtype == torch.float16, f"Input tensor 'a' must be of type torch.float16, but got {a.dtype}"
    assert b.dtype == torch.float16, f"Input tensor 'b' must be of type torch.float16, but got {b.dtype}"
    
    # Make sure the input tensors have the same shape
    assert a.shape == b.shape, f"Input tensors must have the same shape, but got {a.shape} and {b.shape}"
    
    # Get the dimensions of the tensors
    M, N = a.shape
    
    # Allocate output tensor
    c = torch.empty_like(a)
    
    # Define block sizes for the kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    
    # Define the grid for kernel launch
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch the kernel
    vectoradd_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return c

# Example usage to test the implementation
def test_vectoradd():
    # Set tensor size
    N = 1024
    
    # Create input tensors with normal distribution (mean=0, std=1)
    a = torch.randn(N, N, dtype=torch.float16, device='cuda')
    b = torch.randn(N, N, dtype=torch.float16, device='cuda')
    
    # Compute reference result using PyTorch
    ref_output = a + b
    
    # Compute result using our Triton kernel
    triton_output = vectoradd(a, b)
    
    # Verify correctness
    assert torch.allclose(ref_output, triton_output, rtol=1e-2, atol=1e-2), "Results don't match!"
    print("✓ Test passed!")

if __name__ == "__main__":
    test_vectoradd()