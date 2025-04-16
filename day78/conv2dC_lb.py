#!POPCORN leaderboard conv2d

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def conv2d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    batch: tl.constexpr, channels: tl.constexpr, in_h: tl.constexpr, in_w: tl.constexpr,
    ksize: tl.constexpr, out_h: tl.constexpr, out_w: tl.constexpr,
):
    """
    Each program instance computes a single output element at [b, m, y, x].
    Grid dimensions: (batch, channels, out_h*out_w)
    """
    # Get program indices
    b = tl.program_id(0)       # batch index
    m = tl.program_id(1)       # output channel index
    pix_id = tl.program_id(2)  # flattened output pixel index

    # Convert the flattened pixel index into spatial coordinates (y, x)
    y = pix_id // out_w
    x = pix_id % out_w

    acc = 0.0
    # Loop over all input channels and kernel spatial dimensions
    for c in range(channels):
        for i in range(ksize):
            for j in range(ksize):
                # Compute input pointer index for element [b, c, y+i, x+j]
                in_idx = ((b * channels + c) * in_h + (y + i)) * in_w + (x + j)
                # Compute kernel pointer index for element [m, c, i, j]
                ker_idx = ((m * channels + c) * ksize + i) * ksize + j
                a = tl.load(input_ptr + in_idx)
                w = tl.load(kernel_ptr + ker_idx)
                acc += a * w

    # Compute output index for element [b, m, y, x]
    out_idx = ((b * channels + m) * out_h + y) * out_w + x
    tl.store(output_ptr + out_idx, acc)

def custom_kernel(data: input_t) -> output_t:
    """
    Triton implementation of 2D convolution.

    Args:
        data (tuple): A tuple containing:
            - input_tensor: 4D tensor of shape (batch, channels, height, width)
            - kernel: 4D tensor of shape (channels, channels, kernelsize, kernelsize)
    Returns:
        output (torch.Tensor): 4D tensor of shape (batch, channels, height - kernelsize + 1,
                                width - kernelsize + 1) with the convolution results.
    """
    input_tensor, kernel = data
    batch, channels, in_h, in_w = input_tensor.shape
    ksize = kernel.shape[2]
    out_h = in_h - ksize + 1
    out_w = in_w - ksize + 1

    # Ensure tensors are contiguous and on the CUDA device
    input_tensor = input_tensor.contiguous().cuda()
    kernel = kernel.contiguous().cuda()

    output = torch.empty((batch, channels, out_h, out_w), device=input_tensor.device, dtype=torch.float32)

    # Get raw pointers of the tensors (assume float32 data type)
    input_ptr = input_tensor.data_ptr(torch.float32)
    kernel_ptr = kernel.data_ptr(torch.float32)
    output_ptr = output.data_ptr(torch.float32)

    # Define grid dimensions: (batch, channels, out_h*out_w)
    grid = (batch, channels, out_h * out_w)

    # Launch the Triton kernel
    conv2d_kernel[grid](
        input_ptr, kernel_ptr, output_ptr,
        batch, channels, in_h, in_w,
        ksize, out_h, out_w
    )

    return output