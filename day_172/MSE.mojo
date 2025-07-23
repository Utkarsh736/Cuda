from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer
from sync import thread_barrier

const BLOCK_SIZE = 256

@kernel
fn mse_kernel(
    predictions: UnsafePointer[Float32],
    targets: UnsafePointer[Float32],
    partial_sums: UnsafePointer[Float32],
    N: Int32
):
    shared partial: Float32[BLOCK_SIZE]
    let global_id = block_idx.x * block_dim.x + thread_idx.x
    let local_id = thread_idx.x
    let stride = block_dim.x * grid_dim.x

    var sum: Float32 = 0.0
    var i = global_id
    while i < N:
        let diff = predictions[i] - targets[i]
        sum += diff * diff
        i += stride

    partial[local_id] = sum
    thread_barrier()

    # Parallel reduction in shared memory
    var offset = block_dim.x // 2
    while offset > 0:
        if local_id < offset:
            partial[local_id] += partial[local_id + offset]
        thread_barrier()
        offset = offset // 2

    # Store the result of this block
    if local_id == 0:
        partial_sums[block_idx.x] = partial[0]

@export
fn solve(
    predictions: UnsafePointer[Float32],
    targets: UnsafePointer[Float32],
    N: Int32,
    mse: UnsafePointer[Float32]
):
    from gpu.host import DeviceContext

    let grid_size = 1024  # number of blocks
    var ctx = DeviceContext()
    var partial_sums = ctx.alloc_device_memory[Float32](grid_size)

    ctx.enqueue_function[mse_kernel](
        predictions, targets, partial_sums, N,
        grid_dim=(grid_size,), block_dim=(BLOCK_SIZE,)
    )
    ctx.synchronize()

    # Final reduction on CPU
    var host_sums: Float32[grid_size]
    ctx.copy_to_host(partial_sums, &host_sums)

    var total: Float32 = 0.0
    for i in range(grid_size):
        total += host_sums[i]

    mse[0] = total / N
