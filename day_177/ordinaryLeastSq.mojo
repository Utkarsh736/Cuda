# Mojo OLS Solver: Computes β = (Xᵀ X)⁻¹ Xᵀ y
from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from memory import UnsafePointer
from math import ceildiv

# Kernel to compute partial Xᵀ X and Xᵀ y over chunks
fn normal_kernel(
    X: UnsafePointer[Float32],    # shape n_samples × n_features
    y: UnsafePointer[Float32],    # shape n_samples
    partial_gram: UnsafePointer[Float32],  # per-block sums for XᵀX
    partial_Xty: UnsafePointer[Float32],   # per-block sums for Xᵀy
    n_samples: Int32,
    n_features: Int32
):
    let tid = thread_idx.x
    let bid = block_idx.x
    let idx = bid * block_dim.x + tid
    let stride = block_dim.x * grid_dim.x

    # local accumulators
    var local_XX: [Float32](n_features * n_features)
    var local_Xy: [Float32](n_features)
    for i in range(n_features * n_features): local_XX[i] = 0.0
    for f in range(n_features): local_Xy[f] = 0.0

    var i = idx
    while i < n_samples:
        let base = i * n_features
        for r in range(n_features):
            let xr = X[base + r]
            local_Xy[r] += xr * y[i]
            for c in range(n_features):
                local_XX[r * n_features + c] += xr * X[base + c]
        i += stride

    # Write per-block partials
    let gram_off = bid * n_features * n_features
    let xTy_off = bid * n_features
    for i in range(n_features * n_features):
        partial_gram[gram_off + i] = local_XX[i]
    for i in range(n_features):
        partial_Xty[xTy_off + i] = local_Xy[i]

@export
def solve(
    X: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    beta: UnsafePointer[Float32],
    n_samples: Int32,
    n_features: Int32
):
    var ctx = DeviceContext()

    let BLOCK = 256
    let num_blocks = ceildiv(n_samples, BLOCK)

    # Allocate per-block partial results
    let partial_gram = ctx.alloc_device[Float32](num_blocks * n_features * n_features)
    let partial_Xty  = ctx.alloc_device[Float32](num_blocks * n_features)

    ctx.enqueue_function[normal_kernel](
        X, y, partial_gram, partial_Xty, n_samples, n_features,
        grid_dim = num_blocks, block_dim = BLOCK
    )
    ctx.synchronize()

    # Sum over blocks on host
    var gram = [Float32](n_features * n_features)
    var Xty = [Float32](n_features)
    for i in range(n_features * n_features): gram[i] = 0.0
    for i in range(n_features): Xty[i] = 0.0

    var host_gram = ctx.copy_to_host(partial_gram)
    var host_Xty = ctx.copy_to_host(partial_Xty)

    for b in range(num_blocks):
        let g_off = b * n_features * n_features
        let t_off = b * n_features
        for i in range(n_features * n_features):
            gram[i] += host_gram[g_off + i]
        for i in range(n_features):
            Xty[i] += host_Xty[t_off + i]

    # Solve linear system gram * beta = Xty on CPU via Gaussian elimination
    # Basic LU (not pivoted)
    var A = [Float32](n_features * (n_features + 1))
    for i in range(n_features):
        for j in range(n_features):
            A[i * (n_features + 1) + j] = gram[i * n_features + j]
        A[i * (n_features + 1) + n_features] = Xty[i]
    # Forward elimination
    for k in range(n_features):
        let pivot = A[k * (n_features + 1) + k]
        for j in range(k, n_features + 1):
            A[k * (n_features + 1) + j] /= pivot
        for i in range(n_features):
            if i != k:
                let factor = A[i * (n_features + 1) + k]
                for j in range(k, n_features + 1):
                    A[i * (n_features + 1) + j] -= factor * A[k * (n_features + 1) + j]
        A[k * (n_features + 1) + k] = 1.0

    # Extract solution
    for i in range(n_features):
        beta[i] = A[i * (n_features + 1) + n_features]
