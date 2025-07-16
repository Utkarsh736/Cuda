from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

# GPU Bitonic sort kernel (in-place)
fn bitonic_sort_kernel(
    data: UnsafePointer[Float32],
    N: Int32,
    j: Int32,
    k: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= N:
        return

    let ixj = idx ^ j
    if ixj > idx:
        let ascending = ((idx & k) == 0)

        let val1 = data[idx]
        let val2 = data[ixj]

        if (ascending and val1 > val2) or (not ascending and val1 < val2):
            data[idx] = val2
            data[ixj] = val1

# Host launcher for full bitonic sort
@export
def solve(
    data: UnsafePointer[Float32],
    N: Int32
):
    var ctx = DeviceContext()
    var BLOCK_SIZE: Int32 = 256
    let grid_dim = ceildiv(N, BLOCK_SIZE)

    var k = 2
    while k <= N:
        var j = k // 2
        while j > 0:
            ctx.enqueue_function[bitonic_sort_kernel](
                data, N, j, k,
                grid_dim = grid_dim,
                block_dim = BLOCK_SIZE
            )
            ctx.synchronize()
            j //= 2
        k *= 2
