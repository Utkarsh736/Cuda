import numpy as np
from numba import cuda, float32

# Define tile size
TPB = 16  # Threads per block (tile size)

@cuda.jit
def matmul_shared_kernel(A, B, C):
    # Define shared memory arrays for tiles of A and B
    sA = cuda.shared.array((TPB, TPB), dtype=float32)
    sB = cuda.shared.array((TPB, TPB), dtype=float32)

    # Calculate thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TPB + ty
    col = cuda.blockIdx.x * TPB + tx

    # Initialize the accumulation variable
    tmp = 0.0
    n = A.shape[0]  # Assuming square matrices

    # Loop over tiles
    for i in range((n + TPB - 1) // TPB):
        # Load tiles into shared memory
        if row < n and (i * TPB + tx) < n:
            sA[ty, tx] = A[row, i * TPB + tx]
        else:
            sA[ty, tx] = 0.0

        if col < n and (i * TPB + ty) < n:
            sB[ty, tx] = B[i * TPB + ty, col]
        else:
            sB[ty, tx] = 0.0

        # Synchronize to ensure all threads have loaded their data
        cuda.syncthreads()

        # Compute partial product
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Synchronize before loading new tiles
        cuda.syncthreads()

    # Write the result
    if row < n and col < n:
        C[row, col] = tmp

# Host code to test the kernel
n = 1024  # Size of the square matrices
A_host = np.random.rand(n, n).astype(np.float32)
B_host = np.random.rand(n, n).astype(np.float32)
C_host = np.zeros((n, n), dtype=np.float32)

# Transfer data to the device
A_device = cuda.to_device(A_host)
B_device = cuda.to_device(B_host)
C_device = cuda.device_array((n, n), dtype=np.float32)

# Configure the blocks
threads_per_block = (TPB, TPB)
blocks_per_grid = ((n + TPB - 1) // TPB, (n + TPB - 1) // TPB)

# Launch the kernel
matmul_shared_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device)

# Copy the result back to the host
C_device.copy_to_host(C_host)

# Verify the result
print("Result matches NumPy:", np.allclose(C_host, np.dot(A_host, B_host)))
