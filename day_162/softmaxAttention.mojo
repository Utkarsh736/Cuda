from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from memory import UnsafePointer
from math import ceildiv, exp

# Compute dot(Q[i], K[j]) for a row i of Q and all j of K
fn attention_kernel(
    Q: UnsafePointer[Float32],
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    M: Int32, N: Int32, d: Int32
):
    let i = block_idx.x * block_dim.x + thread_idx.x  # Q row index

    if i >= M:
        return

    # Step 1: compute QKᵗ scores (1 × N)
    var scores = [Float32](N)
    var max_score: Float32 = -1e30

    for j in range(N):  # For each K[j]
        var dot: Float32 = 0.0
        for k in range(d):
            dot += Q[i * d + k] * K[j * d + k]
        scores[j] = dot
        if dot > max_score:
            max_score = dot

    # Step 2: apply softmax trick
    var sum_exp: Float32 = 0.0
    for j in range(N):
        scores[j] = exp(scores[j] - max_score)
        sum_exp += scores[j]

    for j in range(N):
        scores[j] /= sum_exp

    # Step 3: output[i] = softmax(QKᵗ) × V
    for k in range(d):
        var val: Float32 = 0.0
        for j in range(N):
            val += scores[j] * V[j * d + k]
        output[i * d + k] = val

# Host launcher
@export
def solve(
    Q: UnsafePointer[Float32],
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    M: Int32, N: Int32, d: Int32
):
    var BLOCK_SIZE: Int32 = 128
    let num_blocks = ceildiv(M, BLOCK_SIZE)

    var ctx = DeviceContext()
    ctx.enqueue_function[attention_kernel](
        Q, K, V, output, M, N, d,
        grid_dim = num_blocks,
        block_dim = BLOCK_SIZE
    )
    ctx.synchronize()
