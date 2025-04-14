import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how to access the next element in a specific dimension
    # E.g., stride_am is the stride to access the next row of A
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A @ B.
    
    Args:
        a_ptr: Pointer to the A matrix
        b_ptr: Pointer to the B matrix
        c_ptr: Pointer to the output C matrix
        M, N, K: Matrix dimensions
        stride_am, stride_ak: Strides for accessing A
        stride_bk, stride_bn: Strides for accessing B
        stride_cm, stride_cn: Strides for accessing C
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile sizes for the GEMM computation
        GROUP_SIZE_M: Number of threads to use for a row of tile
    """
    # Program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Number of tiles in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Number of programs to synchronize (needed for fast barrier)
    num_programs = num_pid_m * num_pid_n
    
    # Get the group ID
    group_id = pid_m // GROUP_SIZE_M
    # Get the ID within the group
    first_pid_m = group_id * GROUP_SIZE_M
    # Compute the range of programs in this group
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Create offsets for the tiles
    # Each program will handle a tile of the output matrix
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks to handle boundary conditions
    a_mask = offs_m[:, None] < M
    b_mask = offs_n[None, :] < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K) * BLOCK_SIZE_K, BLOCK_SIZE_K):
        # Compute the K mask for this tile
        k_mask = (k + offs_k[:, None]) < K
        
        # Load a tile from A (M, K)
        a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak),
                   mask=a_mask[:, None] & k_mask[None, :],
                   other=0.0)
        
        # Load a tile from B (K, N)
        b = tl.load(b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn),
                   mask=k_mask[:, None] & b_mask[None, :],
                   other=0.0)
        
        # Perform the matrix multiplication for this tile
        acc += tl.dot(a, b)
    
    # Write the result to the output matrix
    c_mask = offs_m[:, None] < M and offs_n[None, :] < N
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix multiplication C = A @ B using Triton.
    
    Args:
        a: Input tensor A
        b: Input tensor B
        
    Returns:
        Output tensor C = A @ B
    """
    # Check that the inner dimensions match for matrix multiplication
    assert a.shape[-1] == b.shape[-2], f"Incompatible matrix dimensions: {a.shape} and {b.shape}"
    
    # Extract dimensions
    M, K = a.shape[-2], a.shape[-1]
    K, N = b.shape[-2], b.shape[-1]
    
    # Check that dimensions are multiples of 16 as specified
    assert M % 16 == 0, f"M dimension ({M}) must be a multiple of 16"
    assert N % 16 == 0, f"N dimension ({N}) must be a multiple of 16"
    assert K % 16 == 0, f"K dimension ({K}) must be a multiple of 16"
    
    # Create output tensor
    c = torch.empty((*a.shape[:-1], N), device=a.device, dtype=a.dtype)
    
    # Handle batched matrix multiplication
    if a.dim() > 2 or b.dim() > 2:
        # Compute the batch dimensions
        batch_dims_a = a.shape[:-2]
        batch_dims_b = b.shape[:-2]
        
        # Broadcasting logic
        if batch_dims_a == batch_dims_b:
            # Same batch dimensions, direct broadcasting
            batch_size = torch.prod(torch.tensor(batch_dims_a)).item() if batch_dims_a else 1
            
            # Reshape tensors to 3D for batch processing
            a_3d = a.reshape(-1, M, K)
            b_3d = b.reshape(-1, K, N)
            c_3d = c.reshape(-1, M, N)
            
            # Process each batch
            for i in range(batch_size):
                c_3d[i] = matmul_single(a_3d[i], b_3d[i])
                
            return c
        else:
            # Different batch dimensions, need to handle broadcasting
            # For simplicity, fall back to PyTorch for this case
            return torch.matmul(a, b)
    else:
        # Single matrix multiplication
        return matmul_single(a, b)

def matmul_single(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix multiplication for a single pair of 2D matrices.
    
    Args:
        a: Input tensor A of shape (M, K)
        b: Input tensor B of shape (K, N)
        
    Returns:
        Output tensor C of shape (M, N)
    """
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    
    # Extract dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch the kernel
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c

def custom_matmul(inputs: tuple) -> torch.Tensor:
    """
    Function to handle a tuple of input tensors and apply matmul.
    This is the main entry point matching the leaderboard specification.
    
    Args:
        inputs: A tuple containing two tensors (a, b)
        
    Returns:
        Output tensor resulting from a @ b
    """
    assert isinstance(inputs, tuple) and len(inputs) == 2, "Input must be a tuple of two tensors"
    a, b = inputs
    return matmul(a, b)

# Example usage and testing function
def test_matmul():
    # Test case 1: Simple 2D matrices
    M, N, K = 16, 16, 16  # All multiples of 16
    a = torch.randn((M, K), device='cuda')
    b = torch.randn((K, N), device='cuda')
    
    # Test Triton implementation
    c_triton = custom_matmul((a, b))
    
    # Compare with PyTorch implementation
    c_torch = torch.matmul(a, b)
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-2, atol=1e-2)
    
    # Test case 2: Batched matrices
    batch = 3
    a_batched = torch.randn((batch, M, K), device='cuda')
    b_batched = torch.randn((batch, K, N), device='cuda')
    
    # Test Triton implementation
    c_triton_batched = custom_matmul((a_batched, b_batched))
    
    # Compare with PyTorch implementation
    c_torch_batched = torch.matmul(a_batched, b_batched)
    torch.testing.assert_close(c_triton_batched, c_torch_batched, rtol=1e-2, atol=1e-2)
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_matmul()