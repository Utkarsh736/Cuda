from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

# Kernel: compute one output voxel
fn conv3d_kernel(
    input: UnsafePointer[Float32],
    kernel: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    input_depth: Int32,
    input_rows: Int32,
    input_cols: Int32,
    kernel_depth: Int32,
    kernel_rows: Int32,
    kernel_cols: Int32
):
    let out_d = block_idx.z * block_dim.z + thread_idx.z
    let out_r = block_idx.y * block_dim.y + thread_idx.y
    let out_c = block_idx.x * block_dim.x + thread_idx.x

    let output_depth = input_depth - kernel_depth + 1
    let output_rows = input_rows - kernel_rows + 1
    let output_cols = input_cols - kernel_cols + 1

    if out_d >= output_depth or out_r >= output_rows or out_c >= output_cols:
        return

    var acc: Float32 = 0.0
    for kd in range(kernel_depth):
        for kr in range(kernel_rows):
            for kc in range(kernel_cols):
                let in_d = out_d + kd
                let in_r = out_r + kr
                let in_c = out_c + kc
                let in_idx = (in_d * input_rows + in_r) * input_cols + in_c
                let k_idx = (kd * kernel_rows + kr) * kernel_cols + kc
                acc += input[in_idx] * kernel[k_idx]

    let out_idx = ((out_d * output_rows + out_r) * output_cols) + out_c
    output[out_idx] = acc

@export
def solve(
    input: UnsafePointer[Float32],
    kernel: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    input_depth: Int32,
    input_rows: Int32,
    input_cols: Int32,
    kernel_depth: Int32,
    kernel_rows: Int32,
    kernel_cols: Int32
):
    let output_depth = input_depth - kernel_depth + 1
    let output_rows = input_rows - kernel_rows + 1
    let output_cols = input_cols - kernel_cols + 1

    var BLOCK_D: Int32 = 4
    var BLOCK_Y: Int32 = 8
    var BLOCK_X: Int32 = 8

    let grid_z = ceildiv(output_depth, BLOCK_D)
    let grid_y = ceildiv(output_rows, BLOCK_Y)
    let grid_x = ceildiv(output_cols, BLOCK_X)

    var ctx = DeviceContext()
    ctx.enqueue_function[conv3d_kernel](
        input, kernel, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        grid_dim = (grid_x, grid_y, grid_z),
        block_dim = (BLOCK_X, BLOCK_Y, BLOCK_D)
    )
    ctx.synchronize()
