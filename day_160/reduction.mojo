from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

# Kernel: reduce input into per-block partial sums
fn reduce_kernel(
    input: UnsafePointer[Float32],
    partial_sums: UnsafePointer[Float32],
    N: Int32
):
    var local_sum: Float32 = 0.0
    let tid = thread_idx.x
    let idx = block_idx.x * block_dim.x + tid
    let stride = block_dim.x * grid_dim.x

    var i = idx
    while i < N:
        local_sum += input[i]
        i += stride

    # Shared memory reduction within a block
    __shared__ smem: [Float32, 256]
    smem[tid] = local_sum
    threadgroup.barrier()  # Ensure all threads wrote to shared memory

    var s = block_dim.x // 2
    while s > 0:
        if tid < s:
            smem[tid] += smem[tid + s]
        threadgroup.barrier()
        s //= 2

    if tid == 0:
        partial_sums[block_idx.x] = smem[0]

# Host function
@export
def solve(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int32
):
    var BLOCK_SIZE: Int32 = 256
    let num_blocks = 1024  # fixed or tuned based on device
    let ctx = DeviceContext()

    # Allocate partial sums buffer
    let partial_sums = ctx.alloc_device[Float32](num_blocks)

    # Launch reduction kernel
    ctx.enqueue_function[reduce_kernel](
        input, partial_sums, N,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()

    # Copy partial sums to host and compute final sum on CPU
    var host_sums = ctx.copy_to_host(partial_sums)
    var total: Float32 = 0.0
    for i in range(num_blocks):
        total += host_sums[i]

    output[0] = total
