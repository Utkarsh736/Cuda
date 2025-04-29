#!POPCORN leaderboard conv2d

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def conv2d_kernel(
    input_ptr,    # pointer to input tensor (float32)
    kernel_ptr,   # pointer to filter weights (float32)
    output_ptr,   # pointer to output tensor (float32)
    B, C, H, W,   # batch, channels, height, width
    K,            # kernel size (assume square)
    OutH, OutW,   # output height, width = H-K+1, W-K+1
):
    # 3D grid: (batch, out_channel, pixel_index)
    b = tl.program_id(0)         # batch index 5
    m = tl.program_id(1)         # output channel 6
    pid = tl.program_id(2)       # flattened pixel index (y*OutW + x) 7

    # Compute spatial coordinates
    y = pid // OutW              # output row index 8
    x = pid % OutW               # output column index

    acc = tl.zeros([], dtype=tl.float32)  # accumulator in FP32 9

    # Iterate over input channels and kernel window
    for c in range(C):           # over input channels
        for i in range(K):       # over kernel rows
            for j in range(K):   # over kernel cols
                # Compute input and kernel offsets
                in_idx = ((b * C + c) * H + (y + i)) * W + (x + j)  # row-major 10
                ker_idx = ((m * C + c) * K + i) * K + j             # weight layout 11
                a = tl.load(input_ptr + in_idx)
                w = tl.load(kernel_ptr + ker_idx)
                acc += a * w

    # Write back output
    out_idx = ((b * C + m) * OutH + y) * OutW + x  
    tl.store(output_ptr + out_idx, acc)  # no activation 12

def custom_kernel(data: input_t) -> output_t:
    """
    Triton 2D convolution with valid padding, no stride.
    """
    input_tensor, kernel = data
    B, C, H, W = input_tensor.shape
    _, _, K, _ = kernel.shape
    OutH, OutW = H - K + 1, W - K + 1

    # Ensure contiguous CUDA tensors 13
    input_tensor = input_tensor.contiguous().cuda()
    kernel       = kernel.contiguous().cuda()

    # Allocate output
    output = torch.empty((B, C, OutH, OutW), device=input_tensor.device, dtype=torch.float32)

    # Raw pointers
    input_ptr  = input_tensor.data_ptr()
    kernel_ptr = kernel.data_ptr()
    output_ptr = output.data_ptr()

    # Launch grid: one program per output pixel per channel per batch 14
    grid = (B, C, OutH * OutW)
    conv2d_kernel[grid](
        input_ptr, kernel_ptr, output_ptr,
        B, C, H, W,
        K,
        OutH, OutW
    )
    return output