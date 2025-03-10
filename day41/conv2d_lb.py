#!POPCORN leaderboard conv2d

import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Basic implementation of 2D convolution using PyTorch.

    Args:
        data (tuple[torch.Tensor, torch.Tensor]): 
            - input_tensor: 4D tensor of shape (batch, channels, height, width)
            - kernel: 4D tensor of shape (channels, channels, kernelsize, kernelsize)
    
    Returns:
        torch.Tensor: 4D tensor of shape (batch, channels, height-kernelsize+1, width-kernelsize+1)
    """
    input_tensor, kernel = data  # Unpack input tensors
    return torch.conv2d(input_tensor, kernel, bias=None)  # Perform convolution
