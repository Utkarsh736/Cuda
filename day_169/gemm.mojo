from gpu.id import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from memory import UnsafePointer
from floating_point import convert_fp16_to_fp32, convert_fp32_to_fp16

const BLOCK_SIZE = 16

fn gemm_kernel(
    A: UnsafePointer[Float16],
    B: UnsafePointer[Float16],
    C: UnsafePointer[Float16],
    M: Int32, N: Int32, K: Int32,
    alpha: Float32,
    beta: Float32
):
    row = block_idx.y * block_dim.y + thread_idx.y
    col = block_idx.x * block_dim.x + thread_idx.x

    if row < M and col < N:
        var acc: Float32 = 0.0
        for k in range(K):
            a_val = convert_fp16_to_fp32(A[row * K + k])
            b_val = convert_fp16_to_fp32(B[k * N + col])
            acc += a_val * b_val

        c_val = convert_fp16_to_fp32(C[row * N + col])
        result = alpha * acc + beta * c_val
        C[row * N + col] = convert_fp32_to_fp16(result)

@export
def solve(
    A: UnsafePointer[Float16],
    B: UnsafePointer[Float16],
    C: UnsafePointer[Float16],
    M: Int32,
    N: Int32,
    K: Int32,
    alpha: Float32,
    beta: Float32
):
    var ctx = DeviceContext()
    grid_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (M + BLOCK_SIZE - 1) // BLOCK_SIZE

    ctx.enqueue_function[gemm_kernel](
        A, B, C, M, N, K, alpha, beta,
        grid_dim = (grid_x, grid_y),
        block_dim = (BLOCK_SIZE, BLOCK_SIZE)
    )
    ctx.synchronize()
