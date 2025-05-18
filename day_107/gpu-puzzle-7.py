import numpy as np
from numba import cuda

@cuda.jit
def add_ten_blocks2d(a, out):
    # Compute global 2D indices
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # Guard: only valid (i,j) do work
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = a[i, j] + 10

# Host‑side setup
N, M = 1024, 2048                  # Array dimensions (rows, cols)
a_host = np.arange(N*M, dtype=np.float32).reshape(N, M)

# Device memory
a_dev = cuda.to_device(a_host)
out_dev = cuda.device_array_like(a_host)

# Choose block size smaller than (N, M)
threads_per_block = (16, 16)       # 256 threads per block
blocks_per_grid = (
    (N + threads_per_block[0] - 1) // threads_per_block[0],
    (M + threads_per_block[1] - 1) // threads_per_block[1]
)

# Launch kernel
add_ten_blocks2d[blocks_per_grid, threads_per_block](a_dev, out_dev)

# Retrieve and verify
out_host = out_dev.copy_to_host()
print("Sample [0,:5]:", out_host[0, :5])
print("Sample [-1,-5:]:", out_host[-1, -5:])
