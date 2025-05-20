import numpy as np
from numba import cuda, float32

@cuda.jit
def sum_last3_kernel(a, out):
    # compile‑time constant for threads per block
    TPB = 128
    # allocate shared memory: TPB threads + 2‑wide halo
    sm = cuda.shared.array(shape=TPB + 2, dtype=float32)

    # global index in the input array
    gid = cuda.blockIdx.x * TPB + cuda.threadIdx.x
    # local index into shared buffer (shift by 2 for halo)
    lidx = cuda.threadIdx.x + 2

    # 1) Load “main” element into shared
    if gid < a.size:
        sm[lidx] = a[gid]
    # 2) Threads 0 and 1 load the two‑element halo on the left
    if cuda.threadIdx.x == 0:
        # load a[block_start - 2]
        idx0 = gid - 2
        sm[0] = a[idx0] if idx0 >= 0 else 0.0
        # load a[block_start - 1]
        idx1 = gid - 1
        sm[1] = a[idx1] if idx1 >= 0 else 0.0

    # 3) Synchronize to make sure shared is populated
    cuda.syncthreads()

    # 4) Compute sum of last 3 from shared and write out
    if gid < a.size:
        out[gid] = sm[lidx] + sm[lidx - 1] + sm[lidx - 2]

# Host‑side setup and launch
n = 1024
a_host = np.arange(n, dtype=np.float32)
out_host = np.empty_like(a_host)

# Transfer to device
a_dev = cuda.to_device(a_host)
out_dev = cuda.device_array_like(a_dev)

# Launch with TPB threads per block
threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

sum_last3_kernel[blocks_per_grid, threads_per_block](a_dev, out_dev)

# Retrieve result
out_dev.copy_to_host(out_host)

# Example check
print(out_host[:5])   # [a[0], a[0]+a[1], a[0]+a[1]+a[2], a[1]+a[2]+a[3], …]
