import numpy as np
from numba import cuda

@cuda.jit
def column_sum_kernel(a, out):
    col = cuda.grid(1)
    rows, cols = a.shape
    if col < cols:
        sum_val = 0.0
        for row in range(rows):
            sum_val += a[row, col]
        out[col] = sum_val

# Host-side setup
rows, cols = 1024, 512
a_host = np.random.rand(rows, cols).astype(np.float32)
out_host = np.zeros(cols, dtype=np.float32)

# Transfer to device
a_dev = cuda.to_device(a_host)
out_dev = cuda.device_array(cols, dtype=np.float32)

# Launch configuration
threads_per_block = 128
blocks_per_grid = (cols + threads_per_block - 1) // threads_per_block

# Kernel launch
column_sum_kernel[blocks_per_grid, threads_per_block](a_dev, out_dev)

# Retrieve result
out_host = out_dev.copy_to_host()
print("Column sums:", out_host)
