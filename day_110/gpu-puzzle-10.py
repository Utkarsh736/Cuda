import numpy as np
from numba import cuda

@cuda.jit
def dot_product_kernel(a, b, out):
    idx = cuda.grid(1)                # global thread index
    if idx < a.size:
        # multiply and atomically accumulate into out[0]
        cuda.atomic.add(out, 0, a[idx] * b[idx])

# Host-side setup
n = 1024
a_host = np.arange(n, dtype=np.float32)
b_host = np.arange(n, 0, -1, dtype=np.float32)

# Allocate and initialize output on host
out_host = np.zeros(1, dtype=np.float32)

# Transfer to device
a_dev = cuda.to_device(a_host)
b_dev = cuda.to_device(b_host)
out_dev = cuda.to_device(out_host)

# Launch configuration
threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Kernel launch
dot_product_kernel[blocks_per_grid, threads_per_block](a_dev, b_dev, out_dev)

# Retrieve result
out_dev.copy_to_host(out_host)

print("Dot product:", out_host[0])  # should print sum(a[i]*b[i]) for i in [0,1023]
