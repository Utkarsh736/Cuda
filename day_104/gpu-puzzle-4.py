import numpy as np
from numba import cuda

# Kernel: add 10 to each element of a 2D square array
@cuda.jit
def add_ten_map2d(a, out):
    # 1) Compute 2D global thread indices
    i, j = cuda.grid(2)           # returns (x, y) = threadIdx + blockIdx*blockDim :contentReference[oaicite:4]{index=4}
    # 2) Guard against threads outside the N×N domain
    if i < a.shape[0] and j < a.shape[1]:
        out[i, j] = a[i, j] + 10  # elementwise addition :contentReference[oaicite:5]{index=5}

# Host‑side setup
N = 512                          # dimension of the square array
a_host = np.arange(N*N, dtype=np.float32).reshape(N, N)
# Transfer input to device
a_dev = cuda.to_device(a_host)   # host→device copy :contentReference[oaicite:6]{index=6}
# Allocate output array on device
out_dev = cuda.device_array_like(a_host)  # uninitialized device array :contentReference[oaicite:7]{index=7}

# Launch configuration: 2D blocks and grids with extra threads
threads_per_block = (16, 16)     # e.g., 256 threads per block :contentReference[oaicite:8]{index=8}
blocks_per_grid = (
    (N + threads_per_block[0] - 1) // threads_per_block[0] + 1,
    (N + threads_per_block[1] - 1) // threads_per_block[1] + 1
)                                 # overshoot by one block in each dimension :contentReference[oaicite:9]{index=9}

# Kernel launch
add_ten_map2d[blocks_per_grid, threads_per_block](a_dev, out_dev)  # dispatch 2D grid :contentReference[oaicite:10]{index=10}

# Copy result back to host and verify
out_host = out_dev.copy_to_host()  # device→host copy :contentReference[oaicite:11]{index=11}
print(out_host[0, :5], out_host[-1, -5:])  # e.g., [10, 11, 12, 13, 14] … [N*N-1+10, …]
