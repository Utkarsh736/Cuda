import triton
import triton.language as tl
import torch

@triton.jit
def fp8_matmul_kernel(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Compute pointers for a, b, a_scale, b_scale
        a_offsets = (offs_m[:, None] * stride_am) + ((k + offs_k)[None, :] * stride_ak)
        b_offsets = (offs_n[:, None] * stride_bn) + ((k + offs_k)[None, :] * stride_bk)

        # Load and dequantize a and b
        a_fp8 = tl.load(a_ptr + a_offsets, dtype=tl.float8e4m3fnuz)
        b_fp8 = tl.load(b_ptr + b_offsets, dtype=tl.float8e4m3fnuz)

        a_scale = tl.load(a_scale_ptr + a_offsets)
        b_scale = tl.load(b_scale_ptr + b_offsets)

        a = a_fp8.to(tl.float32) * a_scale
        b = b_fp8.to(tl.float32) * b_scale

        # Accumulate
        acc += tl.dot(a, b.T)

    # Write back to c
    c_offsets = (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptr + c_offsets, acc.to(tl.bfloat16))