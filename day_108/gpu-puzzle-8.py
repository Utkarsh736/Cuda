import numpy as np
from numba import cuda

# Compile‑time constant: size of the shared buffer per block
SHARED_SIZE = 128  # must be a literal constant

@cuda.jit
def add_ten_shared(a, out):
    # Allocate shared memory buffer of fixed size
    shared = cuda.shared.array(shape=SHARED_SIZE, dtype=numba.float32)

    # Compute thread’s global index
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Compute index within the block’s shared buffer
    sidx = cuda.threadIdx.x

    # 1) Load from global memory into shared memory (if in range)
    if tid < a.size and sidx < SHARED_SIZE:
        shared[sidx] = a[tid]
    # 2) Wait until all threads have loaded their element
    cuda.syncthreads()

    # 3) Add 10 in shared memory (only valid entries)
    if tid < a.size and sidx < SHARED_SIZE:
        shared[sidx] += 10

    # 4) Wait until all threads finish updating shared memory
    cuda.syncthreads()

    # 5) Write back from shared memory to global memory
    if tid < a.size and sidx < SHARED_SIZE:
        out[tid] = shared[sidx]

# Host‑side setup
n = 1000
a_host = np.arange(n, dtype=np.float32)
a_dev = cuda.to_device(a_host)
out_dev = cuda.device_array_like(a_host)

# Launch with blocks of SHARED_SIZE threads to exactly fill the shared buffer
threads_per_block = SHARED_SIZE
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Invoke kernel
add_ten_shared[blocks_per_grid, threads_per_block](a_dev, out_dev)

# Retrieve and verify
out_host = out_dev.copy_to_host()
print(out_host[:10], out_host[-5:])
