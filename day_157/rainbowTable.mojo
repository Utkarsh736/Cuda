from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Custom 32-bit integer hash function (deterministic and simple)
fn hash32(x: UInt32) -> UInt32:
    x = ((x >> 16) ^ x) * 0x45d9f3b
    x = ((x >> 16) ^ x) * 0x45d9f3b
    x = (x >> 16) ^ x
    return x

# Kernel to apply R rounds of hashing
fn rainbow_kernel(
    input: UnsafePointer[UInt32],
    output: UnsafePointer[UInt32],
    R: Int32,
    N: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < N:
        var val = input[idx]
        for _ in range(R):
            val = hash32(val)
        output[idx] = val

# Host function to launch the GPU kernel
@export
def solve(
    input: UnsafePointer[UInt32],
    output: UnsafePointer[UInt32],
    R: Int32,
    N: Int32
):
    var BLOCK_SIZE: Int32 = 256
    let num_blocks = ceildiv(N, BLOCK_SIZE)

    var ctx = DeviceContext()
    ctx.enqueue_function[rainbow_kernel](
        input, output, R, N,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
