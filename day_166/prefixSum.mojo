from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

const BLOCK_SIZE = 1024

fn scan_block_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int32
):
    var tid = thread_idx.x
    var gid = block_idx.x * BLOCK_SIZE + tid

    # Shared memory for scan
    var temp: shared [Float32, 2 * BLOCK_SIZE]
    if gid < N:
        temp[tid] = input[gid]
    else:
        temp[tid] = 0.0

    syncthreads()

    # Up-sweep (reduction)
    var offset = 1
    var d = BLOCK_SIZE >> 1
    while d > 0:
        if tid < d:
            let ai = offset * (2 * tid + 1) - 1
            let bi = offset * (2 * tid + 2) - 1
            temp[bi] += temp[ai]
        offset <<= 1
        syncthreads()
        d >>= 1

    # Clear last element (identity)
    if tid == 0:
        temp[BLOCK_SIZE - 1] = 0.0
    syncthreads()

    # Down-sweep (distribution)
    offset = BLOCK_SIZE
    var d2 = 1
    while d2 < BLOCK_SIZE:
        offset >>= 1
        if tid < d2:
            let ai = offset * (2 * tid + 1) - 1
            let bi = offset * (2 * tid + 2) - 1
            let t = temp[ai]
            temp[ai] = temp[bi]
            temp[bi] += t
        syncthreads()
        d2 <<= 1

    if gid < N:
        output[gid] = temp[tid]

# Host launcher
@export
def solve(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int32
):
    var ctx = DeviceContext()
    let num_blocks = ceildiv(N, BLOCK_SIZE)

    ctx.enqueue_function[scan_block_kernel](
        input, output, N,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
