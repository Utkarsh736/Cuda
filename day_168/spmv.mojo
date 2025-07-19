from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer

const BLOCK_SIZE = 256

fn spmv_kernel(
    values: UnsafePointer[Float32],
    col_indices: UnsafePointer[Int32],
    row_ptr: UnsafePointer[Int32],
    x: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    M: Int32
):
    var row = block_idx.x * block_dim.x + thread_idx.x

    if row < M:
        var dot = 0.0
        var row_start = row_ptr[row]
        var row_end = row_ptr[row + 1]

        for i in range(row_start, row_end):
            dot += values[i] * x[col_indices[i]]
        
        y[row] = dot

@export
def solve(
    values: UnsafePointer[Float32],
    col_indices: UnsafePointer[Int32],
    row_ptr: UnsafePointer[Int32],
    x: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    M: Int32
):
    var ctx = DeviceContext()
    let grid_dim = (M + BLOCK_SIZE - 1) // BLOCK_SIZE

    ctx.enqueue_function[spmv_kernel](
        values, col_indices, row_ptr, x, y, M,
        grid_dim = grid_dim,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
