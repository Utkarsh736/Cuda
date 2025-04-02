import numpy as np
import cv2
import triton
import triton.language as tl

@triton.jit
def rgb_to_grayscale_kernel(
    rgb_ptr, gray_ptr,
    H: tl.constexpr, W: tl.constexpr,
    stride_h_rgb, stride_w_rgb, stride_c,
    stride_h_gray, stride_w_gray,
    BLOCK_SIZE: tl.constexpr
):
    # Determine the 2D block indices from the program ID.
    # The grid is laid out in a 1D fashion, so we compute 2D indices.
    num_tiles_per_row = W // BLOCK_SIZE
    pid = tl.program_id(0)
    block_row = pid // num_tiles_per_row
    block_col = pid % num_tiles_per_row

    # Compute the starting indices for this block.
    row_offset = block_row * BLOCK_SIZE
    col_offset = block_col * BLOCK_SIZE

    # Create a range for the block tile.
    rows = row_offset + tl.arange(0, BLOCK_SIZE)
    cols = col_offset + tl.arange(0, BLOCK_SIZE)
    # Use broadcasting to form a grid of (BLOCK_SIZE, BLOCK_SIZE)
    r = tl.reshape(rows, (BLOCK_SIZE, 1))
    c = tl.reshape(cols, (1, BLOCK_SIZE))

    # Compute pointers for each channel
    ptr_r = rgb_ptr + r * stride_h_rgb + c * stride_w_rgb + 0 * stride_c
    ptr_g = rgb_ptr + r * stride_h_rgb + c * stride_w_rgb + 1 * stride_c
    ptr_b = rgb_ptr + r * stride_h_rgb + c * stride_w_rgb + 2 * stride_c

    # Load the values for each channel.
    R = tl.load(ptr_r)
    G = tl.load(ptr_g)
    B = tl.load(ptr_b)

    # Compute grayscale using the standard coefficients.
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    # Compute output pointer for the grayscale image.
    out_ptr = gray_ptr + r * stride_h_gray + c * stride_w_gray
    tl.store(out_ptr, gray)

def rgb_to_grayscale_triton(rgb_image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image (H, W, 3) with values in [0, 1] to a grayscale image (H, W)
    using a Triton kernel.
    """
    H, W, _ = rgb_image.shape

    # Ensure the image is in float32 and contiguous.
    rgb_image = np.ascontiguousarray(rgb_image.astype(np.float32))
    gray = np.empty((H, W), dtype=np.float32)

    # Compute strides in number of elements (not bytes).
    stride_h_rgb = rgb_image.strides[0] // rgb_image.itemsize
    stride_w_rgb = rgb_image.strides[1] // rgb_image.itemsize
    stride_c     = rgb_image.strides[2] // rgb_image.itemsize
    stride_h_gray = gray.strides[0] // gray.itemsize
    stride_w_gray = gray.strides[1] // gray.itemsize

    # Define block size. Note: H and W must be evenly divisible by BLOCK_SIZE.
    BLOCK_SIZE = 32
    num_tiles = (H // BLOCK_SIZE) * (W // BLOCK_SIZE)

    # Launch the Triton kernel.
    rgb_to_grayscale_kernel[grid=(num_tiles,)](
        rgb_image.ctypes.data, gray.ctypes.data,
        H, W,
        stride_h_rgb, stride_w_rgb, stride_c,
        stride_h_gray, stride_w_gray,
        BLOCK_SIZE
    )

    return gray

if __name__ == "__main__":
    # Load an example image using OpenCV (OpenCV loads in BGR order).
    input_image = cv2.imread('input_image.jpg')
    if input_image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Convert to RGB and scale values to [0, 1].
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) / 255.0

    # Convert to grayscale using the Triton kernel.
    gray_image = rgb_to_grayscale_triton(rgb_image)

    # Scale grayscale image back to [0,255] for saving.
    gray_image_uint8 = np.clip(gray_image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite('grayscale_image.jpg', gray_image_uint8)
