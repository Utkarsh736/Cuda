from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim, grid_dim
from memory import UnsafePointer
from math import ceildiv, sqrt

# Kernel: compute next per-agent pos/vel based on neighbor alignment
fn boid_kernel(
    agents: UnsafePointer[Float32],
    agents_next: UnsafePointer[Float32],
    N: Int32,
    radius: Float32,
    dt: Float32
):
    let i = block_idx.x * block_dim.x + thread_idx.x
    if i >= N:
        return

    let xi = agents[4*i]
    let yi = agents[4*i+1]
    let vxi = agents[4*i+2]
    let vyi = agents[4*i+3]

    var sum_vx: Float32 = 0.0
    var sum_vy: Float32 = 0.0
    var count: Int32 = 0

    var j = 0
    while j < N:
        let xj = agents[4*j]
        let yj = agents[4*j+1]
        let dx = xj - xi
        let dy = yj - yi
        if dx*dx + dy*dy <= radius*radius:
            sum_vx += agents[4*j+2]
            sum_vy += agents[4*j+3]
            count += 1
        j += 1

    var nvx = vxi
    var nvy = vyi
    if count > 0:
        let avg_vx = sum_vx / count.to(Float32)
        let avg_vy = sum_vy / count.to(Float32)
        nvx = avg_vx
        nvy = avg_vy

    # update position
    let nx = xi + nvx * dt
    let ny = yi + nvy * dt

    agents_next[4*i]     = nx
    agents_next[4*i + 1] = ny
    agents_next[4*i + 2] = nvx
    agents_next[4*i + 3] = nvy

@export
def solve(
    agents: UnsafePointer[Float32],
    agents_next: UnsafePointer[Float32],
    N: Int32,
    radius: Float32,
    dt: Float32
):
    var ctx = DeviceContext()
    let BLOCK_SIZE: Int32 = 256
    let num_blocks = ceildiv(N, BLOCK_SIZE)

    ctx.enqueue_function[boid_kernel](
        agents, agents_next, N, radius, dt,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
