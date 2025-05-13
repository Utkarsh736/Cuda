import numpy as np
from numba import cuda

@cuda.jit
def add_vectors_kernel(a, b, out):
    idx = cuda.grid(1)               # global 1D thread index :contentReference[oaicite:0]{index=0}
    if idx < a.size:                 # bounds check :contentReference[oaicite:1]{index=1}
        out[idx] = a[idx] + b[idx]   # elementwise add (“zip”) :contentReference[oaicite:2]{index=2}

# Host setup
n = 1024
a_host = np.arange(n, dtype=np.float32)
b_host = np.arange(n, 0, -1, dtype=np.float32)

# Device transfers
a_dev = cuda.to_device(a_host)      # copy input array A :contentReference[oaicite:3]{index=3}
b_dev = cuda.to_device(b_host)      # copy input array B :contentReference[oaicite:4]{index=4}
out_dev = cuda.device_array_like(a_host)

# Launch configuration
threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Kernel launch
add_vectors_kernel[blocks_per_grid, threads_per_block](a_dev, b_dev, out_dev)

# Retrieve result
out_host = out_dev.copy_to_host()   # copy result back :contentReference[oaicite:5]{index=5}

# Verification
print(out_host[:10])  # e.g., [0+1023, 1+1022, …] → [1023., 1023., …]
