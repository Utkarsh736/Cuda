import torch
import triton
import triton.language as tl

@triton.jit
def rgb_to_grayscale_kernel(
    img_ptr,  # pointer to the RGB image [H, W, 3]
    out_ptr,  # pointer to the output grayscale image [H, W]
    height,   # image height
    width,    # image width
    BLOCK_SIZE: tl.constexpr,
):
    """Convert RGB to grayscale using Y = 0.2989 R + 0.5870 G + 0.1140 B"""
    # Program ID and pixel index calculation
    pid = tl.program_id(0)
    num_pixels = height * width
    
    # Each thread processes BLOCK_SIZE pixels (if available)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pixels
    
    # Get 2D coordinates from flat index
    y = offsets // width
    x = offsets % width
    
    # Compute input and output indices
    in_idx = (y * width + x) * 3
    out_idx = y * width + x
    
    # Load RGB values where mask is valid
    r = tl.load(img_ptr + in_idx + 0, mask=mask)
    g = tl.load(img_ptr + in_idx + 1, mask=mask)
    b = tl.load(img_ptr + in_idx + 2, mask=mask)
    
    # Apply grayscale conversion formula
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    # Store result where mask is valid
    tl.store(out_ptr + out_idx, gray, mask=mask)

def rgb_to_grayscale(rgb_img):
    """
    Convert RGB image to grayscale using Triton kernel.
    
    Args:
        rgb_img: PyTorch tensor of shape (H, W, 3) with values in [0, 1]
        
    Returns:
        PyTorch tensor of shape (H, W) with values in [0, 1]
    """
    # Get image dimensions
    height, width, channels = rgb_img.shape
    assert height == width, "Input must be a square image"
    assert height % 2 == 0, "Image dimensions must be even"
    assert channels == 3, "Input must have 3 channels (RGB)"
    
    # Create output tensor
    gray_img = torch.empty((height, width), device=rgb_img.device, dtype=rgb_img.dtype)
    
    # Configure kernel
    BLOCK_SIZE = 128
    grid = (triton.cdiv(height * width, BLOCK_SIZE),)
    
    # Launch kernel
    rgb_to_grayscale_kernel[grid](
        rgb_img.contiguous().data_ptr(),
        gray_img.data_ptr(),
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return gray_img
