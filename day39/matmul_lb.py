#!POPCORN leaderboard matmul

import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Basic implementation of matrix multiplication using PyTorch.

    Args:
        data (tuple[torch.Tensor, torch.Tensor]): Two input tensors with shapes as multiples of 16.
    
    Returns:
        torch.Tensor: Result of matrix multiplication.
    """
    A, B = data  # Unpack input tensors
    return torch.matmul(A, B)  # Perform matrix multiplication
