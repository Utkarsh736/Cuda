#!POPCORN leaderboard vectoradd

import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Basic implementation of float16 vector addition using PyTorch.

    Args:
        data (tuple[torch.Tensor, torch.Tensor]): Two float16 tensors of shape (N, N)
    
    Returns:
        torch.Tensor: A float16 tensor of shape (N, N) containing element-wise sum.
    """
    A, B = data  # Unpack input tensors
    return A + B  # Element-wise addition
