import numpy as np
from numba import cuda

# 1) Define the CUDA kernel with a guard for bounds checking
@cuda.jit
def add_ten_with_guard(a, out):
    idx = cuda.grid(1)            # 1D global thread index :contentReference[oaicite:1]{index=1}
    if idx < a.size:              # Guard to prevent OOB :contentReference[oaicite:2]{index=2}
        out[idx] = a[idx] + 10    # Perform addition only if in bounds :contentReference[oaicite:3]{index=3}

# 2) Host‑side setup
n = 1000
a_host = np.arange(n, dtype=np.float32)              # Create input array :contentReference[oaicite:4]{index=4}
a_dev = cuda.to_device(a_host)                       # Copy to device :contentReference[oaicite:5]{index=5}
out_dev = cuda.device_array_like(a_host)             # Allocate output on device :contentReference[oaicite:6]{index=6}

# 3) Launch configuration with extra threads
threads_per_block = 128
# Intentionally overshoot: add one extra block  
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block + 1  :contentReference[oaicite:7]{index=7}

# 4) Kernel launch
add_ten_with_guard[blocks_per_grid, threads_per_block](a_dev, out_dev)  # Launch with more threads :contentReference[oaicite:8]{index=8}

# 5) Retrieve and verify
out_host = out_dev.copy_to_host()                   # Copy back to host :contentReference[oaicite:9]{index=9}
print(out_host[:10], out_host[-3:])                 # [10,11,…] … [1009,1010,1011]? :contentReference[oaicite:10]{index=10}
