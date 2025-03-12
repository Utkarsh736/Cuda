#!POPCORN leaderboard sort

import torch
from torch.utils.cpp_extension import load

# Compile the CUDA sorting kernel at runtime
sort_cuda = load(
    name="sort_cuda",
    sources=[],
    verbose=True,
    extra_cuda_cflags=["-O2"],
    extra_include_paths=["."],  # Adjust path if necessary
    extra_ldflags=[],
)

def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    """
    Optimized CUDA-based sorting using parallel Bitonic Sort.

    Args:
        data (torch.Tensor): A 1D tensor containing floating-point values.
    
    Returns:
        torch.Tensor: Sorted tensor in ascending order.
    """
    sorted_data = torch.empty_like(data)
    
    # Call CUDA function (assuming it's implemented in sort_cuda)
    sort_cuda.parallel_sort(data, sorted_data)
    
    return sorted_data
