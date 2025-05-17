import numpy as np
from numba import cuda

@cuda.jit
def add_ten_blocks(a, out):
    # Compute global 1D index from block and thread IDs
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # Guard against out‑of‑bounds
    if idx < a.size:
        out[idx] = a[idx] + 10

# Host‑side setup
n = 10000
a_host = np.arange(n, dtype=np.float32)

# Copy to device and allocate output
a_dev = cuda.to_device(a_host)
out_dev = cuda.device_array_like(a_host)

# Choose fewer threads per block than n
threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Launch kernel
add_ten_blocks[blocks_per_grid, threads_per_block](a_dev, out_dev)

# Retrieve and verify
out_host = out_dev.copy_to_host()
print(out_host[:10], out_host[-5:])
