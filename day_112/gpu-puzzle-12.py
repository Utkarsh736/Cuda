import numpy as np
from numba import cuda, float32

@cuda.jit
def block_prefix_sum(a, out):
    # Shared memory size = threads per block
    sm = cuda.shared.array(128, dtype=float32)

    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gid = bid * bdim + tid

    # Load from global memory into shared memory
    if gid < a.size:
        sm[tid] = a[gid]
    else:
        sm[tid] = 0.0

    cuda.syncthreads()

    # Parallel reduction (tree-based sum)
    offset = 1
    while offset < bdim:
        if tid >= offset:
            val = sm[tid] + sm[tid - offset]
        else:
            val = sm[tid]
        cuda.syncthreads()
        sm[tid] = val
        cuda.syncthreads()
        offset *= 2

    # After full reduction, last thread has the block sum
    if tid == bdim - 1:
        out[bid] = sm[tid]

# Host setup
n = 1024
a_host = np.random.rand(n).astype(np.float32)
threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

a_dev = cuda.to_device(a_host)
out_dev = cuda.device_array(shape=(blocks_per_grid,), dtype=np.float32)

# Kernel launch
block_prefix_sum[blocks_per_grid, threads_per_block](a_dev, out_dev)

# Retrieve results
out_host = out_dev.copy_to_host()
print("Block-wise sums:", out_host)
