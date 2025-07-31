from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

const BLOCK = 16  # block dimension for tiled multiplication

# Kernel: C = A × B, all size N×N
fn matmul_kernel(
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    C: UnsafePointer[Float32],
    N: Int32
):
    let row = block_idx.y * block_dim.y + thread_idx.y
    let col = block_idx.x * block_dim.x + thread_idx.x
    if row >= N or col >= N:
        return

    var sum: Float32 = 0.0
    for k in range(N):
        sum += A[row * N + k] * B[k * N + col]
    C[row * N + col] = sum

@export
def solve(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int32,
    P: Int32
):
    var ctx = DeviceContext()

    # allocate two temp matrices
    let temp1 = ctx.alloc_device[Float32](N * N)
    let temp2 = ctx.alloc_device[Float32](N * N)

    # init temp1 = input (base), output = identity if P=0
    ctx.enqueue_function[copy_kernel](
        input, temp1, N * N,
        grid_dim = ceildiv(N*N, BLOCK*BLOCK),
        block_dim = (BLOCK*BLOCK,)
    )
    ctx.synchronize()

    if P == 0:
        # identity
        ctx.enqueue_function[identity_kernel](output, N)
        ctx.synchronize()
        return

    # initialize output = temp1 if P>=1
    ctx.enqueue_function[copy_kernel](
        temp1, output, N * N,
        grid_dim = ceildiv(N*N, BLOCK*BLOCK),
        block_dim = (BLOCK*BLOCK,)
    )
    ctx.synchronize()

    var cur_power = 1

    while cur_power < P:
        # output = output * temp1 -> temp2
        ctx.enqueue_function[matmul_kernel](
            output, temp1, temp2, N,
            grid_dim = (ceildiv(N, BLOCK), ceildiv(N, BLOCK)),
            block_dim = (BLOCK, BLOCK)
        )
        ctx.synchronize()

        # swap buffers
        let swap = output
        output = temp2
        temp2 = swap

        cur_power += 1
    }
