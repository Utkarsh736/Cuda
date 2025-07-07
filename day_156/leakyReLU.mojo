from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Leaky ReLU coefficient
let alpha: Float32 = 0.01

# Kernel to apply Leaky ReLU
fn leaky_relu_kernel(
    x: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    N: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < N:
        let val = x[idx]
        y[idx] = val if val >= 0 else val * alpha

# Host function
@export
def solve(
    x: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    N: Int32
):
    var BLOCK_SIZE: Int32 = 256
    let num_blocks = ceildiv(N, BLOCK_SIZE)

    var ctx = DeviceContext()
    ctx.enqueue_function[leaky_relu_kernel](
        x, y, N,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
