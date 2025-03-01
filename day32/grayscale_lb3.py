#!POPCORN leaderboard grayscale

# This is a submission template for popcorn leaderboard 'grayscale'.
# Your task is as follows:
# > Implement an RGB to grayscale conversion kernel that matches the reference implementation.

# > The kernel should convert square RGB images with even sizes to grayscale using the standard coefficients:

# > Y = 0.2989 R + 0.5870 G + 0.1140 B

# > 

# > Input: RGB tensor of shape (H, W, 3) with values in [0, 1]

# > Output: Grayscale tensor of shape (H, W) with values in [0, 1]

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

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
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move input to the appropriate device
    data = data.to(device)
    
    # Extract RGB channels
    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]
    
    # Apply grayscale conversion using standard coefficients
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    # Ensure the output is on the CPU as required
    return grayscale.to('cpu')
