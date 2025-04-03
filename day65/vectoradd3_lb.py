import torch
import triton
import triton.language as tl

@triton.jit
def mat_add_kernel(
    a_ptr, b_ptr, c_ptr,
    N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_m = offset_m < N
    mask_n = offset_n < N

    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_n[None, :] * stride_an
    b_ptrs = b_ptr + offset_m[:, None] * stride_bm + offset_n[None, :] * stride_bn
    c_ptrs = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn

    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    c = a + b

    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])

def mat_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float16
    N = a.shape[0]
    assert N == a.shape[1], "Input tensors must be square matrices"

    c = torch.empty_like(a)

    grid = (triton.cdiv(N, 16), triton.cdiv(N, 16))

    mat_add_kernel[grid](
        a, b, c,
        N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE=16
    )

    return c

# Example usage
N = 128
a = torch.randn((N, N), dtype=torch.float16, device='cuda')
b = torch.randn((N, N), dtype=torch.float16, device='cuda')

c = mat_add(a, b)
