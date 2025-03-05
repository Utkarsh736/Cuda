#!POPCORN leaderboard histogram

import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Basic implementation of histogram computation in PyTorch.
    
    Args:
        data (torch.Tensor): Tensor of shape (size,) containing values in [0, 100]
    
    Returns:
        torch.Tensor: Histogram tensor of shape (num_bins,)
    """
    size = data.shape[0]
    num_bins = size // 16  # Number of bins based on input size

    # Compute histogram using PyTorch
    hist = torch.histc(data, bins=num_bins, min=0, max=100)

    return hist
