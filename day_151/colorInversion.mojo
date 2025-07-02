from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

# Kernel to invert colors in-place (R, G, B channels); Alpha remains unchanged
fn invert_kernel(
    image: UnsafePointer[UInt8],   # 1D array of RGBA values
    width: Int32, height: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    let total_pixels = width * height

    if idx < total_pixels:
        let base = idx * 4
        image[base + 0] = 255 - image[base + 0]  # R
        image[base + 1] = 255 - image[base + 1]  # G
        image[base + 2] = 255 - image[base + 2]  # B
        # image[base + 3] is Alpha, unchanged

# Host function
@export
def solve(
    image: UnsafePointer[UInt8],  # In-place RGBA buffer
    width: Int32, height: Int32
):
    var BLOCK_SIZE: Int32 = 256
    let total_pixels = width * height
    let num_blocks = ceildiv(total_pixels, BLOCK_SIZE)

    var ctx = DeviceContext()
    ctx.enqueue_function[invert_kernel](
        image, width, height,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
