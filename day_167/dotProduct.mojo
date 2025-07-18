from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

const BLOCK_SIZE = 1024

fn dot_product_kernel(
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    partial_sums: UnsafePointer[Float32],
    N: Int32
):
    var tid = thread_idx.x
    var i = block_idx.x * BLOCK_SIZE + tid

    var sum = 0.0
    if i < N:
        sum = A[i] * B[i]

    # Shared memory reduction
    var cache: shared [Float32, BLOCK_SIZE]
    cache[tid] = sum
    syncthreads()

    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            cache[tid] += cache[tid + stride]
        syncthreads()
        stride //= 2

    # Write block's result
    if tid == 0:
        partial_sums[block_idx.x] = cache[0]

@export
def solve(
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int32
):
    var ctx = DeviceContext()
    let blocks = ceildiv(N, BLOCK_SIZE)

    var partial_sums = ctx.alloc_device[Float32](blocks)

    ctx.enqueue_function[dot_product_kernel](
        A, B, partial_sums, N,
        grid_dim = blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()

    # Reduce partial results on CPU
    var final_sum = 0.0
    for i in range(blocks):
        final_sum += partial_sums[i]

    output[0] = final_sum
