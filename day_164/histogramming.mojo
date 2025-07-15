from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer
from math import ceildiv
from atomic import atomic_add

# Kernel: builds histogram using atomic adds
fn histogram_kernel(
    input: UnsafePointer[Int32],
    histogram: UnsafePointer[Int32],
    N: Int32,
    num_bins: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    let stride = block_dim.x * grid_dim.x

    var i = idx
    while i < N:
        let val = input[i]
        if 0 <= val and val < num_bins:
            atomic_add(histogram + val, 1)
        i += stride

# Host function
@export
def solve(
    input: UnsafePointer[Int32],
    histogram: UnsafePointer[Int32],
    N: Int32,
    num_bins: Int32
):
    var BLOCK_SIZE: Int32 = 256
    let num_blocks = ceildiv(N, BLOCK_SIZE)

    var ctx = DeviceContext()

    # Zero-initialize histogram (device-side)
    ctx.memset(histogram, 0, num_bins * Int32.sizeof())

    # Launch histogram kernel
    ctx.enqueue_function[histogram_kernel](
        input, histogram, N, num_bins,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
