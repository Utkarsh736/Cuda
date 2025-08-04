from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

const BLOCK = 256

# 1. GPU kernel for assignment: assign each point to nearest centroid
fn assign_kernel(
    data_x: UnsafePointer[Float32],
    data_y: UnsafePointer[Float32],
    cent_x: UnsafePointer[Float32],
    cent_y: UnsafePointer[Float32],
    labels: UnsafePointer[Int32],
    N: Int32,
    k: Int32
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= N:
        return

    let px = data_x[idx]
    let py = data_y[idx]

    var best = 0
    var best_dist = (px - cent_x[0])*(px - cent_x[0]) + (py - cent_y[0])*(py - cent_y[0])
    for c in range(1, k):
        let dx = px - cent_x[c]
        let dy = py - cent_y[c]
        let d = dx*dx + dy*dy
        if d < best_dist:
            best = c
            best_dist = d

    labels[idx] = best

@export
def solve(
    data_x: UnsafePointer[Float32],
    data_y: UnsafePointer[Float32],
    sample_size: Int32,
    k: Int32,
    max_iterations: Int32,
    initial_centroid_x: UnsafePointer[Float32],
    initial_centroid_y: UnsafePointer[Float32],
    labels: UnsafePointer[Int32],
    final_centroid_x: UnsafePointer[Float32],
    final_centroid_y: UnsafePointer[Float32]
):
    var ctx = DeviceContext()

    # Host-centroids arrays
    var cent_x = [Float32](k)
    var cent_y = [Float32](k)

    # Initialize centroids
    for c in range(k):
        cent_x[c] = initial_centroid_x[c]
        cent_y[c] = initial_centroid_y[c]

    let num_blocks = ceildiv(sample_size, BLOCK)

    for iter in range(max_iterations):
        # 1. assign each point to nearest centroid on GPU
        ctx.copy_to_device(cent_x, cent_x, k)
        ctx.copy_to_device(cent_y, cent_y, k)
        ctx.enqueue_function[assign_kernel](
            data_x, data_y, cent_x, cent_y, labels,
            sample_size, k,
            grid_dim = num_blocks, block_dim = BLOCK
        )
        ctx.synchronize()

        # 2. accumulate per-cluster sums on host
        var sum_x = [Float32](k)
        var sum_y = [Float32](k)
        var count = [Int32](k)
        for c in range(k):
            sum_x[c] = 0.0
            sum_y[c] = 0.0
            count[c] = 0

        # copy labels back to host
        var host_labels = ctx.copy_to_host(labels)

        for i in range(sample_size):
            let c = host_labels[i]
            sum_x[c] += data_x[i]
            sum_y[c] += data_y[i]
            count[c] += 1

        var changed = false

        for c in range(k):
            if count[c] > 0:
                let nx = sum_x[c] / count[c].to(Float32)
                let ny = sum_y[c] / count[c].to(Float32)
                if nx != cent_x[c] or ny != cent_y[c]:
                    changed = true
                cent_x[c] = nx
                cent_y[c] = ny

        # stop if converged
        if not changed:
            break

    # write final centroids to output
    for c in range(k):
        final_centroid_x[c] = cent_x[c]
        final_centroid_y[c] = cent_y[c]
