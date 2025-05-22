import numpy as np
from numba import cuda

@cuda.jit
def conv1d_kernel(a, b, out):
    i = cuda.grid(1)              # global thread index
    n = a.size
    m = b.size
    # full convolution length = n + m - 1
    if i < n + m - 1:
        s = 0.0
        # sum over kernel elements
        for j in range(m):
            ai = i - j
            if 0 <= ai < n:
                s += a[ai] * b[j]
        out[i] = s              # one global write

# Host-side setup
n = 1024
m = 7
a_host = np.random.rand(n).astype(np.float32)
b_host = np.random.rand(m).astype(np.float32)
out_host = np.zeros(n + m - 1, dtype=np.float32)

# Transfer to device
a_dev = cuda.to_device(a_host)
b_dev = cuda.to_device(b_host)
out_dev = cuda.to_device(out_host)

# Launch configuration
threads_per_block = 128
blocks_per_grid = (out_host.size + threads_per_block - 1) // threads_per_block

# Kernel launch
conv1d_kernel[blocks_per_grid, threads_per_block](a_dev, b_dev, out_dev)

# Retrieve result
out_dev.copy_to_host(out_host)

# Verify against NumPy
expected = np.convolve(a_host, b_host)
print("Match:", np.allclose(out_host, expected))
