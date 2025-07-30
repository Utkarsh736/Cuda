from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

const BITS_PER_PASS = 8
const BUCKETS = 1 << BITS_PER_PASS
const MASK = BUCKETS - 1

# Kernel: count bucket occurrences per block
fn count_kernel(
    input: UnsafePointer[UInt32],
    counts: UnsafePointer[Int32],
    N: Int32,
    shift: Int32
):
    let tid = thread_idx.x
    let bid = block_idx.x
    let idx = bid * block_dim.x + tid
    let stride = block_dim.x * grid_dim.x

    var local_counts: [Int32](BUCKETS)
    for b in range(BUCKETS): local_counts[b] = 0

    var i = idx
    while i < N:
        let key = (input[i] >> shift) & MASK
        local_counts[key] += 1
        i += stride

    # Write per-block local counts
    for b in range(BUCKETS):
        counts[bid * BUCKETS + b] = local_counts[b]

# Kernel: scatter elements into output using prefix sums
fn scatter_kernel(
    input: UnsafePointer[UInt32],
    output: UnsafePointer[UInt32],
    global_offsets: UnsafePointer[Int32],
    N: Int32,
    shift: Int32
):
    let tid = thread_idx.x
    let idx = block_idx.x * block_dim.x + tid
    let stride = block_dim.x * grid_dim.x

    var local_offsets = [Int32](BUCKETS)
    # copy global offsets
    for b in range(BUCKETS):
        local_offsets[b] = global_offsets[b]

    var i = idx
    while i < N:
        let key = (input[i] >> shift) & MASK
        let dest = atomic_add(&local_offsets[key], 1)
        output[dest] = input[i]
        i += stride

@export
def solve(
    input: UnsafePointer[UInt32],
    output: UnsafePointer[UInt32],
    N: Int32
):
    var ctx = DeviceContext()

    let BLOCK = 256
    let num_blocks = ceildiv(N, BLOCK)
    # Temporary buffers for counts
    let counts_buf = ctx.alloc_device[Int32]([num_blocks * BUCKETS])
    var temp_buf = ctx.alloc_device[UInt32]([N])

    var in_buf = input
    var out_buf = temp_buf

    for pass in range(0, 32, BITS_PER_PASS):
        # 1. count phase
        ctx.enqueue_function[count_kernel](
            in_buf, counts_buf, N, pass,
            grid_dim = num_blocks, block_dim = BLOCK
        )
        ctx.synchronize()

        # 2. aggregate counts to compute global prefix sums
        var host_counts = ctx.copy_to_host(counts_buf)
        var global_offsets = [Int32](BUCKETS)
        var bucket_sums = [Int32](BUCKETS)
        for b in range(BUCKETS):
            bucket_sums[b] = 0
        for blk in range(num_blocks):
            for key in range(BUCKETS):
                let v = host_counts[blk * BUCKETS + key]
                bucket_sums[key] += v
        # prefix sum
        var sum = 0
        for key in range(BUCKETS):
            global_offsets[key] = sum
            sum += bucket_sums[key]

        # 3. scatter phase
        let d_offsets = ctx.alloc_device[Int32]([BUCKETS])
        ctx.enqueue_function[scatter_kernel](
            in_buf, out_buf, d_offsets, N, pass,
            grid_dim = num_blocks, block_dim = BLOCK
        )
        # copy offsets
        ctx.copy_to_device(global_offsets, d_offsets)
        ctx.synchronize()

        # swap buffers
        let tmp = in_buf
        in_buf = out_buf
        out_buf = tmp

    # if final is in temp, copy back
    if in_buf != output:
        ctx.copy_to_device(in_buf, output, N)
