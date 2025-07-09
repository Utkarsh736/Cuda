from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# GPU kernel to copy N x N matrix A to matrix B (row-major format)
fn matrix_copy_kernel(
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    N: Int32
):
    let row = block_idx.y * block_dim.y + thread_idx.y
    let col = block_idx.x * block_dim.x + thread_idx.x

    if row < N and col < N:
        let idx = row * N + col
        B[idx] = A[idx]

# Host launcher
@export
def solve(
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    N: Int32
):
    var BLOCK_SIZE_X: Int32 = 16
    var BLOCK_SIZE_Y: Int32 = 16

    let grid_dim_x = ceildiv(N, BLOCK_SIZE_X)
    let grid_dim_y = ceildiv(N, BLOCK_SIZE_Y)

    var ctx = DeviceContext()
    ctx.enqueue_function[matrix_copy_kernel](
        A, B, N,
        grid_dim = (grid_dim_x, grid_dim_y),
        block_dim = (BLOCK_SIZE_X, BLOCK_SIZE_Y)
    )
    ctx.synchronize()
