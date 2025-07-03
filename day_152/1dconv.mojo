from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Kernel to perform 1D valid convolution
fn conv1d_kernel(
    input: UnsafePointer[Float32],
    kernel: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    input_size: Int32,
    kernel_size: Int32
):
    let out_idx = block_idx.x * block_dim.x + thread_idx.x
    let output_size = input_size - kernel_size + 1

    if out_idx < output_size:
        var sum: Float32 = 0.0
        for k in range(kernel_size):
            sum += input[out_idx + k] * kernel[kernel_size - 1 - k]
        output[out_idx] = sum

# Host function to launch the kernel
@export
def solve(
    input: UnsafePointer[Float32],
    kernel: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    input_size: Int32,
    kernel_size: Int32
):
    var BLOCK_SIZE: Int32 = 256
    let output_size = input_size - kernel_size + 1
    let num_blocks = ceildiv(output_size, BLOCK_SIZE)

    var ctx = DeviceContext()
    ctx.enqueue_function[conv1d_kernel](
        input, kernel, output, input_size, kernel_size,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
