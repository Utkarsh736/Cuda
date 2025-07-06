from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# GPU kernel to reverse array elements in-place
fn reverse_kernel(
    input: UnsafePointer[Float32],
    N: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    let mid = N // 2

    if idx < mid:
        let j = N - 1 - idx
        let tmp = input[idx]
        input[idx] = input[j]
        input[j] = tmp

# Host function to launch the kernel
@export
def solve(
    input: UnsafePointer[Float32],
    N: Int32
):
    var BLOCK_SIZE: Int32 = 256
    let num_swaps = N // 2
    let num_blocks = ceildiv(num_swaps, BLOCK_SIZE)

    var ctx = DeviceContext()
    ctx.enqueue_function[reverse_kernel](
        input, N,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
