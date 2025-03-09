#!POPCORN leaderboard sort

import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Basic implementation of sorting using PyTorch.

    Args:
        data (torch.Tensor): A 1D tensor containing flattened floating-point values.
    
    Returns:
        torch.Tensor: Sorted tensor in ascending order.
    """
    return torch.sort(data)[0]  # Perform sorting and return sorted values
