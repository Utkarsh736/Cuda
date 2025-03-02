#!POPCORN leaderboard grayscale

from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
    """
    Convert RGB image to grayscale using standard coefficients:
    Y = 0.2989 R + 0.5870 G + 0.1140 B
    
    Args:
        data (input_t): RGB tensor of shape (H, W, 3) with values in [0, 1]
        
    Returns:
        output_t: Grayscale tensor of shape (H, W) with values in [0, 1]
    """
    # Extract RGB channels (preserving the device)
    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]
    
    # Apply grayscale conversion using standard coefficients
    # This keeps the result on the same device as the input data
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    # No need to move the tensor - keep it on the original device
    return grayscale
