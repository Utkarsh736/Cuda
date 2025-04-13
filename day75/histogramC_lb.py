import torch
import triton
import triton.language as tl

@triton.jit
def histogram_kernel(
    # Pointers to input data and output histogram
    data_ptr,
    hist_ptr,
    # Size of the input data
    data_size,
    # Number of bins for the histogram
    num_bins,
    # Range of the histogram
    min_range: tl.constexpr,
    max_range: tl.constexpr,
    # Block size for processing
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing a histogram of input data.
    
    Arguments:
        data_ptr: Pointer to the input data tensor
        hist_ptr: Pointer to the output histogram tensor
        data_size: Number of elements in the input data
        num_bins: Number of bins in the histogram
        min_range: Minimum value of the histogram range
        max_range: Maximum value of the histogram range
        BLOCK_SIZE: Number of elements to process per block
    """
    # Program ID (block index)
    pid = tl.program_id(axis=0)
    
    # Compute the block start offset
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the elements this block will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where the block extends beyond the input size
    mask = offsets < data_size
    
    # Load the elements for this block
    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate the bin width
    bin_width = (max_range - min_range) / num_bins
    
    # Calculate bin indices for each element
    bin_indices = tl.minimum(
        num_bins - 1,  # Cap at num_bins - 1 to handle edge case
        tl.maximum(
            0,  # Ensure no negative indices
            tl.floor((data - min_range) / bin_width).to(tl.int32)
        )
    )
    
    # Iterate through each possible bin index
    for bin_idx in range(num_bins):
        # Count elements that fall into this bin
        count = tl.sum(tl.where(bin_indices == bin_idx, 1, 0) & mask)
        
        # Atomically add the count to the appropriate bin in the histogram
        if count > 0:
            tl.atomic_add(hist_ptr + bin_idx, count)

def compute_histogram(data: torch.Tensor) -> torch.Tensor:
    """
    Compute a histogram of the input data using Triton.
    
    Arguments:
        data: Input tensor of shape (size,)
        
    Returns:
        Histogram tensor of shape (num_bins,)
    """
    # Ensure input is a 1D tensor
    assert data.dim() == 1, f"Input tensor must be 1-dimensional, but got shape {data.shape}"
    
    # Check that size is a multiple of 16
    data_size = data.numel()
    assert data_size % 16 == 0, f"Input size ({data_size}) must be a multiple of 16"
    
    # Calculate number of bins (size / 16)
    num_bins = data_size // 16
    
    # Define range for histogram
    min_range = 0.0
    max_range = 100.0
    
    # Define block size for the kernel
    BLOCK_SIZE = 1024
    
    # Initialize output histogram tensor with zeros
    histogram = torch.zeros(num_bins, dtype=torch.int32, device=data.device)
    
    # Calculate grid size based on input size and block size
    grid = (triton.cdiv(data_size, BLOCK_SIZE),)
    
    # Launch the histogram kernel
    histogram_kernel[grid](
        data_ptr=data,
        hist_ptr=histogram,
        data_size=data_size,
        num_bins=num_bins,
        min_range=min_range,
        max_range=max_range,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return histogram

# Alternative implementation using per-thread histograms for better performance
@triton.jit
def histogram_kernel_optimized(
    # Pointers to input data and output histogram
    data_ptr,
    hist_ptr,
    # Size of the input data
    data_size,
    # Number of bins for the histogram
    num_bins,
    # Range of the histogram
    min_range: tl.constexpr,
    max_range: tl.constexpr,
    # Block size for processing
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for computing a histogram of input data.
    This version uses shared memory for per-thread local histograms to reduce atomic contention.
    
    Arguments:
        data_ptr: Pointer to the input data tensor
        hist_ptr: Pointer to the output histogram tensor
        data_size: Number of elements in the input data
        num_bins: Number of bins in the histogram
        min_range: Minimum value of the histogram range
        max_range: Maximum value of the histogram range
        BLOCK_SIZE: Number of elements to process per block
    """
    # Program ID (block index)
    pid = tl.program_id(axis=0)
    
    # Compute the block start offset
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the elements this block will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where the block extends beyond the input size
    mask = offsets < data_size
    
    # Load the elements for this block
    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate the bin width
    bin_width = (max_range - min_range) / num_bins
    
    # Calculate bin indices for each element
    bin_indices = tl.minimum(
        num_bins - 1,  # Cap at num_bins - 1 to handle edge case
        tl.maximum(
            0,  # Ensure no negative indices
            tl.floor((data - min_range) / bin_width).to(tl.int32)
        )
    )
    
    # Create a local histogram in shared memory
    local_hist = tl.zeros([num_bins], dtype=tl.int32)
    
    # Populate the local histogram
    for i in range(BLOCK_SIZE):
        if i < tl.shape(data)[0] and mask[i]:
            bin_idx = bin_indices[i]
            local_hist = tl.atomic_add(local_hist, [bin_idx], 1)[0]
    
    # Synchronize threads
    tl.debug_barrier()
    
    # Atomically add local histogram counts to the global histogram
    for bin_idx in range(num_bins):
        count = local_hist[bin_idx]
        if count > 0:
            tl.atomic_add(hist_ptr + bin_idx, count)

def compute_histogram_optimized(data: torch.Tensor) -> torch.Tensor:
    """
    Compute a histogram of the input data using an optimized Triton kernel.
    
    Arguments:
        data: Input tensor of shape (size,)
        
    Returns:
        Histogram tensor of shape (num_bins,)
    """
    # Ensure input is a 1D tensor
    assert data.dim() == 1, f"Input tensor must be 1-dimensional, but got shape {data.shape}"
    
    # Check that size is a multiple of 16
    data_size = data.numel()
    assert data_size % 16 == 0, f"Input size ({data_size}) must be a multiple of 16"
    
    # Calculate number of bins (size / 16)
    num_bins = data_size // 16
    
    # Define range for histogram
    min_range = 0.0
    max_range = 100.0
    
    # Define block size for the kernel
    BLOCK_SIZE = 1024
    
    # Initialize output histogram tensor with zeros
    histogram = torch.zeros(num_bins, dtype=torch.int32, device=data.device)
    
    # Calculate grid size based on input size and block size
    grid = (triton.cdiv(data_size, BLOCK_SIZE),)
    
    # Launch the histogram kernel
    histogram_kernel_optimized[grid](
        data_ptr=data,
        hist_ptr=histogram,
        data_size=data_size,
        num_bins=num_bins,
        min_range=min_range,
        max_range=max_range,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return histogram

# Example usage to test the implementations
def test_histogram():
    # Set up a test case where size is a multiple of 16
    size = 1024  # This will give 64 bins (1024/16)
    
    # Create input tensor with values between 0 and 100
    data = torch.rand(size, device='cuda') * 100
    
    # Compute reference histogram using PyTorch
    num_bins = size // 16
    bin_edges = torch.linspace(0, 100, num_bins + 1, device='cuda')
    ref_hist = torch.histc(data, bins=num_bins, min=0, max=100).to(torch.int32)
    
    # Compute histogram using our Triton kernel
    triton_hist = compute_histogram(data)
    
    # Compute histogram using optimized Triton kernel
    triton_hist_optimized = compute_histogram_optimized(data)
    
    # Verify correctness - count should match for each bin
    assert torch.all(ref_hist == triton_hist), "Histograms don't match!"
    assert torch.all(ref_hist == triton_hist_optimized), "Optimized histograms don't match!"
    
    print("✓ Test passed!")
    print(f"Number of bins: {num_bins}")
    print(f"Input size: {size}")
    
    # Print sample of the histogram results
    sample_size = min(10, num_bins)
    print(f"\nFirst {sample_size} bins:")
    for i in range(sample_size):
        print(f"Bin {i}: PyTorch = {ref_hist[i].item()}, Triton = {triton_hist[i].item()}, Optimized = {triton_hist_optimized[i].item()}")

if __name__ == "__main__":
    test_histogram()