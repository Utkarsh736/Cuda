import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def rgb_to_grayscale_kernel(
    rgb_ptr,
    gray_ptr,
    H,
    W,
    RGB_STRIDE,
    GRAY_STRIDE,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Kernel for RGB to grayscale conversion."""
    h_offset = tl.program_id(0) * BLOCK_SIZE_H
    w_offset = tl.program_id(1) * BLOCK_SIZE_W

    h_mask = (h_offset + tl.arange(0, BLOCK_SIZE_H)) < H
    w_mask = (w_offset + tl.arange(0, BLOCK_SIZE_W)) < W

    h_indices = h_offset + tl.arange(0, BLOCK_SIZE_H)[:, None]
    w_indices = w_offset + tl.arange(0, BLOCK_SIZE_W)[None, :]

    rgb_indices = h_indices[:, :, None] * RGB_STRIDE + w_indices[:, :, None] * 3

    r_ptr = rgb_ptr + rgb_indices + 0
    g_ptr = rgb_ptr + rgb_indices + 1
    b_ptr = rgb_ptr + rgb_indices + 2

    r = tl.load(r_ptr, mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    g = tl.load(g_ptr, mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    b = tl.load(b_ptr, mask=h_mask[:, None] & w_mask[None, :], other=0.0)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    gray_ptr = gray_ptr + h_indices * GRAY_STRIDE + w_indices

    tl.store(gray_ptr, gray, mask=h_mask[:, None] & w_mask[None, :])


def rgb_to_grayscale(rgb_tensor):
    """Wrapper function for RGB to grayscale conversion."""
    H, W, _ = rgb_tensor.shape
    gray_tensor = torch.zeros((H, W), dtype=rgb_tensor.dtype, device=rgb_tensor.device)

    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32

    grid = (triton.cdiv(H, BLOCK_SIZE_H), triton.cdiv(W, BLOCK_SIZE_W))

    rgb_to_grayscale_kernel[grid](
        rgb_tensor,
        gray_tensor,
        H,
        W,
        rgb_tensor.stride(0),
        gray_tensor.stride(0),
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )

    return gray_tensor

def reference_rgb_to_grayscale(rgb_tensor):
    """Reference RGB to grayscale conversion using PyTorch."""
    r, g, b = rgb_tensor.unbind(dim=-1)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def test_rgb_to_grayscale():
    """Test function for RGB to grayscale conversion."""
    H = 256
    W = 256
    rgb_tensor = torch.rand((H, W, 3), dtype=torch.float32, device="cuda")

    gray_triton = rgb_to_grayscale(rgb_tensor)
    gray_ref = reference_rgb_to_grayscale(rgb_tensor)

    torch.testing.assert_close(gray_triton, gray_ref)
    print("RGB to grayscale conversion test passed!")

if __name__ == "__main__":
    test_rgb_to_grayscale()
