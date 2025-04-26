import triton
import triton.language as tl

@triton.jit
def prefix_sum_blockwise(
    x_ptr,          # pointer to input
    out_ptr,        # pointer to output
    sum_ptr,        # pointer to block sums
    n_elements,     # total N
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)                                   # block index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)       # global indices
    mask = offs < n_elements                                  # bounds mask

    # Load into registers
    x = tl.load(x_ptr + offs, mask=mask)                     # 0

    # Inclusive scan in shared memory (Blelloch up-sweep & down-sweep)
    # Here we use a simple Hillis–Steele scan
    for offset in range(1, BLOCK_SIZE):
        y = tl.shift(x, -offset, mask=mask, other=0.0)       # load x[i-offset]
        x = tl.where(offs >= offset, x + y, x)
        tl.barrier()                                         # 1

    # Write partial results
    tl.store(out_ptr + offs, x, mask=mask)

    # Thread 0 writes the block sum
    if tl.arange(0, BLOCK_SIZE)[0] == 0:
        last = x[mask].max()                                 # last valid element
        tl.store(sum_ptr + pid, last)                       # 2