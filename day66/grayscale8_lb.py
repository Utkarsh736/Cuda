import torch
import triton
import triton.language as tl

@triton.jit
def rgb_to_grayscale_kernel(
    # Pointers to matrices
    img_ptr,  # pointer to the RGB image [H, W, 3]
    out_ptr,  # pointer to the output grayscale image [H, W]
    # Matrix dimensions
    height,   # image height
    width,    # image width
    # Meta-parameters
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    """
    Converts an RGB image to grayscale using the formula:
    Y = 0.2989 R + 0.5870 G + 0.1140 B
    
    Uses Triton's block-level parallelism for efficient GPU execution.
    """
    # Program ID
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    # Calculate starting pixel coordinates for this block
    start_x = pid_x * BLOCK_SIZE_X
    start_y = pid_y * BLOCK_SIZE_Y
    
    # Calculate offsets for each thread in the block
    offs_x = start_x + tl.arange(0, BLOCK_SIZE_X)
    offs_y = start_y + tl.arange(0, BLOCK_SIZE_Y)
    
    # Create a mask to handle edge cases
    mask_x = offs_x < width
    mask_y = offs_y < height
    
    # Process each pixel in the block
    for y in range(BLOCK_SIZE_Y):
        for x in range(BLOCK_SIZE_X):
            # Check if current pixel is within image boundaries
            if mask_x[x] & mask_y[y]:
                # Calculate pixel index in the flattened arrays
                idx_in = (offs_y[y] * width + offs_x[x]) * 3
                idx_out = offs_y[y] * width + offs_x[x]
                
                # Load RGB values for the current pixel
                r = tl.load(img_ptr + idx_in)
                g = tl.load(img_ptr + idx_in + 1)
                b = tl.load(img_ptr + idx_in + 2)
                
                # Apply grayscale conversion formula
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                
                # Store result
                tl.store(out_ptr + idx_out, gray)

def rgb_to_grayscale(rgb_image):
    """
    Convert RGB image to grayscale using Triton kernel.
    
    Args:
        rgb_image: PyTorch tensor of shape (H, W, 3) with values in [0, 1]
        
    Returns:
        PyTorch tensor of shape (H, W) with values in [0, 1]
    """
    # Ensure input is a square image with even dimensions
    height, width, channels = rgb_image.shape
    assert height == width, "Input must be a square image"
    assert height % 2 == 0, "Image dimensions must be even"
    assert channels == 3, "Input must have 3 channels (RGB)"
    assert rgb_image.is_cuda, "Input tensor must be on GPU"
    
    # Create output tensor
    gray_image = torch.empty((height, width), device=rgb_image.device, dtype=rgb_image.dtype)
    
    # Define block sizes for the kernel
    # Choose block sizes that are divisible by warp size (32) for optimal performance
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32
    
    # Calculate grid dimensions
    grid_x = triton.cdiv(width, BLOCK_SIZE_X)
    grid_y = triton.cdiv(height, BLOCK_SIZE_Y)
    
    # Launch kernel
    rgb_to_grayscale_kernel[(grid_x, grid_y)](
        rgb_image.contiguous().data_ptr(),
        gray_image.data_ptr(),
        height,
        width,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return gray_image

# Test function to validate against reference implementation
def test_rgb_to_grayscale():
    """
    Test the Triton implementation against PyTorch reference implementation.
    """
    # Create a random RGB image (on CPU first)
    height = width = 256  # Square image with even dimensions
    rgb_image = torch.rand((height, width, 3), dtype=torch.float32)
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("Warning: CUDA not available, test will run on CPU")
    rgb_image = rgb_image.to(device)
    
    # Reference implementation
    r, g, b = rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]
    reference = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    # Triton implementation
    result = rgb_to_grayscale(rgb_image)
    
    # Validate results
    max_diff = torch.max(torch.abs(reference - result)).item()
    print(f"Maximum difference: {max_diff}")
    assert max_diff < 1e-5, "Results differ significantly from reference"
    print("Test passed!")

if __name__ == "__main__":
    # Test the implementation
    test_rgb_to_grayscale()
