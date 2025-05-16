import numpy as np
from numba import cuda

@cuda.jit
def broadcast_add_kernel(a, b, out):
    idx = cuda.grid(1)                  # 1D global thread index
    if idx < a.size:                    # guard: only valid a‑indices
        # wrap b’s index to broadcast its values if b is shorter than a
        bj = idx % b.size
        out[idx] = a[idx] + b[bj]

# Host‑side setup
n = 1024
m = 10                                  # length of b (can be < n)
a_host = np.arange(n, dtype=np.float32)
b_host = np.linspace(0, 9, m, dtype=np.float32)

# Transfer to device
a_dev = cuda.to_device(a_host)
b_dev = cuda.to_device(b_host)
out_dev = cuda.device_array_like(a_host)

# Launch with extra threads
threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block + 1

broadcast_add_kernel[blocks_per_grid, threads_per_block](a_dev, b_dev, out_dev)

# Retrieve and verify
out_host = out_dev.copy_to_host()
print(out_host[:20])   # first 20 results: a[i] + b[i%10]
