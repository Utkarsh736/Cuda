from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from memory import UnsafePointer
from math import ceildiv, exp

# --------- Step 1: Find max value ---------
fn max_kernel(
    input: UnsafePointer[Float32],
    partial_max: UnsafePointer[Float32],
    N: Int32
):
    let tid = thread_idx.x
    let idx = block_idx.x * block_dim.x + tid
    let stride = block_dim.x * grid_dim.x

    var local_max: Float32 = -1e30
    var i = idx
    while i < N:
        let val = input[i]
        if val > local_max:
            local_max = val
        i += stride

    __shared__ smem: [Float32, 256]
    smem[tid] = local_max
    threadgroup.barrier()

    var s = block_dim.x // 2
    while s > 0:
        if tid < s:
            if smem[tid + s] > smem[tid]:
                smem[tid] = smem[tid + s]
        threadgroup.barrier()
        s //= 2

    if tid == 0:
        partial_max[block_idx.x] = smem[0]

# --------- Step 2: Compute exp(x - max) and reduce sum ---------
fn exp_sum_kernel(
    input: UnsafePointer[Float32],
    exp_values: UnsafePointer[Float32],
    partial_sum: UnsafePointer[Float32],
    max_val: Float32,
    N: Int32
):
    let tid = thread_idx.x
    let idx = block_idx.x * block_dim.x + tid
    let stride = block_dim.x * grid_dim.x

    var local_sum: Float32 = 0.0
    var i = idx
    while i < N:
        let e = exp(input[i] - max_val)
        exp_values[i] = e
        local_sum += e
        i += stride

    __shared__ smem: [Float32, 256]
    smem[tid] = local_sum
    threadgroup.barrier()

    var s = block_dim.x // 2
    while s > 0:
        if tid < s:
            smem[tid] += smem[tid + s]
        threadgroup.barrier()
        s //= 2

    if tid == 0:
        partial_sum[block_idx.x] = smem[0]

# --------- Step 3: Normalize ---------
fn normalize_kernel(
    exp_values: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    total_sum: Float32,
    N: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < N:
        output[idx] = exp_values[idx] / total_sum

# --------- Host Function ---------
@export
def solve(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int32
):
    var BLOCK_SIZE: Int32 = 256
    let num_blocks = ceildiv(N, BLOCK_SIZE)

    var ctx = DeviceContext()

    # Step 1: find max
    let partial_max = ctx.alloc_device[Float32](num_blocks)
    ctx.enqueue_function[max_kernel](
        input, partial_max, N,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
    var host_max = ctx.copy_to_host(partial_max)
    var max_val: Float32 = -1e30
    for i in range(num_blocks):
        if host_max[i] > max_val:
            max_val = host_max[i]

    # Step 2: compute exp and sum
    let exp_values = ctx.alloc_device[Float32](N)
    let partial_sum = ctx.alloc_device[Float32](num_blocks)
    ctx.enqueue_function[exp_sum_kernel](
        input, exp_values, partial_sum, max_val, N,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
    var host_sum = ctx.copy_to_host(partial_sum)
    var total_sum: Float32 = 0.0
    for i in range(num_blocks):
        total_sum += host_sum[i]

    # Step 3: normalize
    ctx.enqueue_function[normalize_kernel](
        exp_values, output, total_sum, N,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
