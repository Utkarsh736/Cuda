#!POPCORN leaderboard amd-fp8-mm

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def fp8_mm_kernel(
    a_ptr, b_ptr,
    a_scale_ptr, b_scale_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k

        a_fp8 = tl.load(a_ptr + offs_m[:, None] * stride_am + k_ids[None, :])
        b_fp8 = tl.load(b_ptr + offs_n[:, None] * stride_bn + k_ids[None, :])  # column-major B

        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * stride_am + k_ids[None, :])
        b_scale = tl.load(b_scale_ptr + offs_n[:, None] * stride_bn + k_ids[None, :])

        # Decode e4m3fnuz (simulate as int8 for now, Triton lacks native e4m3fp8)
        a = a_fp8.to(tl.float32) * a_scale
        b = b_fp8.to(tl.float32) * b_scale

        # B is column-major, so need to transpose it
        a_acc += tl.dot(a, b, allow_tf32=False)

    # Write result as row-major BF16
    c_offset = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr + c_offset, a_acc.to(tl.bfloat16))


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    M, K = a.shape
    N = b.shape[0]  # b is N x K column-major

    # Setup strides
    stride_am, stride_ak = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_cn = c.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fp8_mm_kernel[grid](
        a, b, a_scale, b_scale, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    return c