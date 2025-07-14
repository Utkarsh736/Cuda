from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

# Kernel for valid 2D convolution
fn conv2d_kernel(
    input: UnsafePointer[Float32],
    kernel: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    input_rows: Int32,
    input_cols: Int32,
    kernel_rows: Int32,
    kernel_cols: Int32
):
    let out_row = block_idx.y * block_dim.y + thread_idx.y
    let out_col = block_idx.x * block_dim.x + thread_idx.x

    let output_rows = input_rows - kernel_rows + 1
    let output_cols = input_cols - kernel_cols + 1

    if out_row >= output_rows or out_col >= output_cols:
        return

    var acc: Float32 = 0.0
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            let in_r = out_row + i
            let in_c = out_col + j
            let input_idx = in_r * input_cols + in_c
            let kernel_idx = i * kernel_cols + j
            acc += input[input_idx] * kernel[kernel_idx]

    let output_idx = out_row * output_cols + out_col
    output[output_idx] = acc

# Host function
@export
def solve(
    input: UnsafePointer[Float32],
    kernel: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    input_rows: Int32,
    input_cols: Int32,
    kernel_rows: Int32,
    kernel_cols: Int32
):
    let output_rows = input_rows - kernel_rows + 1
    let output_cols = input_cols - kernel_cols + 1

    var BLOCK_X: Int32 = 16
    var BLOCK_Y: Int32 = 16

    let grid_x = ceildiv(output_cols, BLOCK_X)
    let grid_y = ceildiv(output_rows, BLOCK_Y)

    var ctx = DeviceContext()
    ctx.enqueue_function[conv2d_kernel](
        input, kernel, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols,
        grid_dim = (grid_x, grid_y),
        block_dim = (BLOCK_X, BLOCK_Y)
    )
    ctx.synchronize()
