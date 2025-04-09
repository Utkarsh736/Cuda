import torch
import triton
import triton.language as tl

@triton.jit
def grayscale_kernel(
    input_ptr,       # Pointer to the input tensor (RGB image)
    output_ptr,      # Pointer to the output tensor (grayscale image)
    H,               # Height of the image
    W,               # Width of the image
    stride_yh,       # Stride to jump between rows in the input tensor
    stride_xw,       # Stride to jump between columns in the input tensor
    stride_c,        # Stride to jump between channels in the input tensor
    stride_out_h,    # Stride to jump between rows in the output tensor
    stride_out_w,    # Stride to jump between columns in the output tensor
    BLOCK_SIZE: tl.constexpr, # Use a compile-time constant for block size
):
    """
    Triton kernel for converting an RGB image to grayscale.

    Each instance of this kernel processes a BLOCK_SIZE x BLOCK_SIZE tile
    of the output image.
    """
    # 1. Calculate the pixel coordinates this program instance will handle
    # Get the program ID for the x and y dimensions (block indices)
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Create ranges for pixel offsets within the block
    offs_x = tl.arange(0, BLOCK_SIZE)
    offs_y = tl.arange(0, BLOCK_SIZE)

    # Calculate the absolute pixel coordinates for the block
    # Shape: (BLOCK_SIZE,)
    x_px = pid_x * BLOCK_SIZE + offs_x
    y_px = pid_y * BLOCK_SIZE + offs_y

    # Create 2D coordinates for loading/storing blocks
    # offs_y[:, None] -> Shape (BLOCK_SIZE, 1)
    # offs_x[None, :] -> Shape (1, BLOCK_SIZE)
    # Broadcasting results in y_px_2d and x_px_2d having shape (BLOCK_SIZE, BLOCK_SIZE)
    y_px_2d = y_px[:, None]
    x_px_2d = x_px[None, :]

    # 2. Calculate memory offsets for input tensor (RGB)
    # Base offset for the top-left corner of the block
    # input_ptr is the start of the tensor data
    # Add offsets for y-coordinate (rows), x-coordinate (columns), and channel
    # Shape: (BLOCK_SIZE, BLOCK_SIZE)
    r_offs = input_ptr + y_px_2d * stride_yh + x_px_2d * stride_xw + 0 * stride_c
    g_offs = input_ptr + y_px_2d * stride_yh + x_px_2d * stride_xw + 1 * stride_c
    b_offs = input_ptr + y_px_2d * stride_yh + x_px_2d * stride_xw + 2 * stride_c

    # 3. Create a boundary mask
    # This mask prevents loading/storing data outside the actual image dimensions
    # Shape: (BLOCK_SIZE, BLOCK_SIZE)
    boundary_mask = (y_px_2d < H) & (x_px_2d < W)

    # 4. Load RGB values safely using the mask
    # If the mask is False for a pixel, tl.load will load 0 (or another specified value)
    # This avoids out-of-bounds memory access
    # Shape: (BLOCK_SIZE, BLOCK_SIZE)
    r = tl.load(r_offs, mask=boundary_mask)
    g = tl.load(g_offs, mask=boundary_mask)
    b = tl.load(b_offs, mask=boundary_mask)

    # 5. Apply the grayscale conversion formula
    # Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    # Calculation is done element-wise on the loaded blocks
    # Shape: (BLOCK_SIZE, BLOCK_SIZE)
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # 6. Calculate memory offsets for the output tensor (Grayscale)
    # Shape: (BLOCK_SIZE, BLOCK_SIZE)
    output_offs = output_ptr + y_px_2d * stride_out_h + x_px_2d * stride_out_w

    # 7. Store the result safely using the mask
    # Only writes results for pixels within the image boundaries
    tl.store(output_offs, grayscale, mask=boundary_mask)


def custom_kernel(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton grayscale kernel.

    Args:
        input_tensor: A PyTorch tensor of shape (H, W, 3) representing the RGB image,
                      with values in [0, 1]. H=W and both are even.

    Returns:
        A PyTorch tensor of shape (H, W) representing the grayscale image,
        with values in [0, 1].
    """
    # Basic input validation (optional but good practice)
    if not (input_tensor.ndim == 3 and input_tensor.shape[2] == 3):
        raise ValueError("Input tensor must have shape (H, W, 3)")
    if not input_tensor.is_cuda:
        raise TypeError("Input tensor must be a CUDA tensor")
    if input_tensor.dtype != torch.float32:
         # Kernels often optimized for float32, ensure consistency
         # Or handle different types if necessary
        input_tensor = input_tensor.to(torch.float32)
        # print("Warning: Input tensor converted to float32")

    H, W, C = input_tensor.shape

    # Create the output tensor with the desired shape (H, W)
    # Ensure it's on the same device and has a compatible dtype
    output_tensor = torch.empty((H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    # Define the block size for the kernel
    BLOCK_SIZE = 32

    # Define the grid dimensions
    # Use ceiling division to ensure all pixels are covered
    grid = lambda meta: (
        triton.cdiv(W, meta['BLOCK_SIZE']), # Number of blocks in X dimension
        triton.cdiv(H, meta['BLOCK_SIZE']), # Number of blocks in Y dimension
    )

    # Launch the kernel
    grayscale_kernel[grid](
        input_tensor,       # Pass the tensor directly (Triton handles pointer extraction)
        output_tensor,
        H,
        W,
        input_tensor.stride(0),  # Stride for dim 0 (Height)
        input_tensor.stride(1),  # Stride for dim 1 (Width)
        input_tensor.stride(2),  # Stride for dim 2 (Channel)
        output_tensor.stride(0), # Stride for dim 0 (Height) of output
        output_tensor.stride(1), # Stride for dim 1 (Width) of output
        BLOCK_SIZE=BLOCK_SIZE,   # Pass BLOCK_SIZE as a compile-time constant
        # num_warps=4 # Optional: can tune for performance
    )

    return output_tensor

# Example Usage (for testing locally)
if __name__ == '__main__':
    # Create a sample RGB image tensor on GPU
    height, width = 256, 256 # Example even dimensions H=W
    # Ensure dtype is float32 as often expected by kernels and for [0,1] range
    rgb_image = torch.rand(height, width, 3, device='cuda', dtype=torch.float32)

    # Reference CPU implementation for verification
    def reference_grayscale(img_rgb):
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return img_gray

    # Run the Triton kernel
    grayscale_triton = custom_kernel(rgb_image)

    # Run the reference implementation
    grayscale_ref = reference_grayscale(rgb_image.cpu()) # Move to CPU for numpy-like op

    # Compare results
    print("Triton output shape:", grayscale_triton.shape)
    print("Reference output shape:", grayscale_ref.shape)

    # Check for correctness (allowing for small floating point differences)
    # Move triton result to CPU for comparison
    print("Are results close?", torch.allclose(grayscale_triton.cpu(), grayscale_ref, atol=1e-5))

    # You can add timing comparisons here as well using torch.cuda.Event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm-up
    for _ in range(10):
        custom_kernel(rgb_image)

    # Time the kernel
    start_event.record()
    for _ in range(100):
         custom_kernel(rgb_image)
    end_event.record()

    # Waits for everything to finish and calculates time
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Triton kernel average execution time: {elapsed_time_ms / 100:.6f} ms")
