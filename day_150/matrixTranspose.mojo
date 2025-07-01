from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Kernel to transpose a matrix A (rows x cols) -> output (cols x rows)
fn transpose_kernel(
    A: UnsafePointer[Float32],          # Input matrix: rows x cols
    output: UnsafePointer[Float32],     # Output matrix: cols x rows (transpose)
    rows: Int32, cols: Int32
):
    let row = block_idx.y * block_dim.y + thread_idx.y
    let col = block_idx.x * block_dim.x + thread_idx.x

    if row < rows and col < cols:
        let input_idx = row * cols + col
        let output_idx = col * rows + row
        output[output_idx] = A[input_idx]

# Host launcher function
@export
def solve(
    A: UnsafePointer[Float32],            # Input matrix (rows x cols)
    output: UnsafePointer[Float32],       # Output matrix (cols x rows)
    rows: Int32, cols: Int32
):
    var BLOCK_SIZE_X: Int32 = 16
    var BLOCK_SIZE_Y: Int32 = 16

    var grid_dim_x = ceildiv(cols, BLOCK_SIZE_X)
    var grid_dim_y = ceildiv(rows, BLOCK_SIZE_Y)

    var ctx = DeviceContext()
    ctx.enqueue_function[transpose_kernel](
        A, output, rows, cols,
        grid_dim = (grid_dim_x, grid_dim_y),
        block_dim = (BLOCK_SIZE_X, BLOCK_SIZE_Y)
    )
    ctx.synchronize()
