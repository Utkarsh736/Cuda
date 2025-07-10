from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Kernel to compute partial sums of y_samples
fn monte_carlo_partial_sum_kernel(
    y_samples: UnsafePointer[Float32],
    partial_sums: UnsafePointer[Float32],
    n_samples: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    let stride = block_dim.x * grid_dim.x

    var sum: Float32 = 0.0
    var i = idx
    while i < n_samples:
        sum += y_samples[i]
        i += stride

    # Store per-thread partial sum (1 sum per thread)
    partial_sums[idx] = sum

# Host function
@export
def solve(
    y_samples: UnsafePointer[Float32],
    a: Float32,
    b: Float32,
    n_samples: Int32,
    result: UnsafePointer[Float32]
):
    var BLOCK_SIZE: Int32 = 256
    let total_threads = 1024 * BLOCK_SIZE  # Large number of threads for better utilization
    let num_blocks = ceildiv(total_threads, BLOCK_SIZE)

    # Allocate device-side memory for partial sums (1 per thread)
    var ctx = DeviceContext()
    let partial_sums = ctx.alloc_device[Float32](total_threads)

    # Launch kernel to compute partial sums
    ctx.enqueue_function[monte_carlo_partial_sum_kernel](
        y_samples, partial_sums, n_samples,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()

    # Copy partial_sums back and compute total sum on CPU
    var host_sums = ctx.copy_to_host(partial_sums)
    var total_sum: Float32 = 0.0
    for i in range(total_threads):
        total_sum += host_sums[i]

    result[0] = (b - a) * total_sum / n_samples
