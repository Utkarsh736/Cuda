import torch
import triton
import triton.language as tl

@triton.jit
def rgb_to_grayscale(
    input_ptr,
    output_ptr,
    H,
    W,
    n_elements,
    strides_input_h,
    strides_input_w,
    strides_input_c,
    strides_output_h,
    strides_output_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = offs % W
    y = offs // W
    r_offset = y * strides_input_h + x * strides_input_w + 0 * strides_input_c
    g_offset = y * strides_input_h + x * strides_input_w + 1 * strides_input_c
    b_offset = y * strides_input_h + x * strides_input_w + 2 * strides_input_c
    r = tl.load(input_ptr + r_offset)
    g = tl.load(input_ptr + g_offset)
    b = tl.load(input_ptr + b_offset)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    output_offset = y * strides_output_h + x * strides_output_w
    tl.store(output_ptr + output_offset, gray)

def custom_kernel(input: torch.Tensor) -> torch.Tensor:
    H, W, _ = input.shape
    n_elements = H * W
    output = torch.empty((H, W), device=input.device, dtype=input.dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta['block_size']),)
    rgb_to_grayscale[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        H=H,
        W=W,
        n_elements=n_elements,
        strides_input_h=input.stride(0),
        strides_input_w=input.stride(1),
        strides_input_c=input.stride(2),
        strides_output_h=output.stride(0),
        strides_output_w=output.stride(1),
        BLOCK_SIZE=32
    )
    return output
