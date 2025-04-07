import torch
import triton
import triton.language as tl

@triton.jit
def histogram_kernel(
    data_ptr,      # pointer to input data
    hist_ptr,      # pointer to output histogram
    n_elements,    # number of elements in input
    n_bins,        # number of histogram bins
    min_range,     # minimum value of range (0)
    max_range,     # maximum value of range (100)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute histogram of input data using atomic adds.
    Bins are evenly spaced between min_range and max_range.
    """
    # Program ID and block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate offsets and mask for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data elements from input
    elements = tl.load(data_ptr + offsets, mask=mask)
    
    # Calculate bin width and bin indices
    bin_width = (max_range - min_range) / n_bins
    
    # Process each element in this block
    for i in range(BLOCK_SIZE):
        if i < offsets.shape[0] and mask[i]:
            # Get the current element
            element = elements[i]
            
            # Check if element is within range
            if element >= min_range and element < max_range:
                # Calculate bin index
                bin_idx = tl.math.floor((element - min_range) / bin_width)
                
                # Clamp bin index to valid range (necessary for edge cases)
                bin_idx = tl.math.min(bin_idx, n_bins - 1)
                
                # Update histogram with atomic add
                tl.atomic_add(hist_ptr + bin_idx, 1)

def compute_histogram(data):
    """
    Compute histogram of input data across the range 0 to 100.
    
    Args:
        data: PyTorch tensor of shape (size,)
        
    Returns:
        PyTorch tensor of shape (n_bins,) containing bin counts
    """
    # Input validation
    assert data.dim() == 1, "Input must be a 1D tensor"
    assert data.shape[0] % 16 == 0, "Input size must be a multiple of 16"
    
    # Define range
    min_range = 0.0
    max_range = 100.0
    
    # Calculate number of bins (size / 16)
    n_elements = data.shape[0]
    n_bins = n_elements // 16
    
    # Ensure data is on GPU and contiguous
    data = data.contiguous()
    
    # Create output histogram tensor initialized to zeros
    histogram = torch.zeros(n_bins, device=data.device, dtype=torch.int32)
    
    # Configure kernel
    BLOCK_SIZE = 1024  # Process 1024 elements per block
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    histogram_kernel[grid](
        data.data_ptr(),
        histogram.data_ptr(),
        n_elements,
        n_bins,
        min_range,
        max_range,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return histogram
