#!POPCORN leaderboard prefix_sum

import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Basic implementation of inclusive prefix sum (scan) using PyTorch.

    Args:
        data (torch.Tensor): A 1D tensor of size n
    
    Returns:
        torch.Tensor: A 1D tensor of size n containing the inclusive prefix sum.
    """
    return torch.cumsum(data, dim=0)  # Compute the inclusive scan (prefix sum)
