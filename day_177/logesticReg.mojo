from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer
from math import exp, ceildiv

const LR: Float32 = 0.01    # learning rate
const MAX_ITERS = 1000

# GPU kernel to compute gradient contributions
fn grad_kernel(
    X: UnsafePointer[Float32],     # shape n_samples × n_features
    y: UnsafePointer[Float32],     # shape n_samples
    beta: UnsafePointer[Float32],  # current parameters
    grad_out: UnsafePointer[Float32], # per-block gradient sums
    n_samples: Int32,
    n_features: Int32
):
    let tid = thread_idx.x
    let bid = block_idx.x
    let idx = bid * block_dim.x + tid
    let stride = block_dim.x * grid_dim.x

    # allocate per-thread local gradient
    var local_grad: [Float32](n_features)
    for j in range(n_features):
        local_grad[j] = 0.0

    var i = idx
    while i < n_samples:
        # compute dot product wᵀx_i
        let base = i * n_features
        var dot: Float32 = 0.0
        for j in range(n_features):
            dot += beta[j] * X[base + j]
        let pred = 1.0 / (1.0 + exp(-dot))
        let err = pred - y[i]
        for j in range(n_features):
            local_grad[j] += err * X[base + j]
        i += stride

    # accumulate thread's local_grad into per-block memory (direct store)
    let out_off = bid * n_features
    for j in range(n_features):
        grad_out[out_off + j] = local_grad[j]

@export
def solve(
    X: UnsafePointer[Float32],
    y: UnsafePointer[Float32],
    beta: UnsafePointer[Float32],
    n_samples: Int32,
    n_features: Int32
):
    var ctx = DeviceContext()
    # initialize beta to zeros
    for j in range(n_features):
        beta[j] = 0.0

    let BLOCK = 256
    let num_blocks = ceildiv(n_samples, BLOCK)

    # allocate per-block gradients
    var grad_out = ctx.alloc_device[Float32](num_blocks * n_features)

    for iter in range(MAX_ITERS):
        # launch GPU kernel to compute gradients
        ctx.enqueue_function[grad_kernel](
            X, y, beta, grad_out, n_samples, n_features,
            grid_dim = num_blocks, block_dim = BLOCK
        )
        ctx.synchronize()

        # copy gradients to host and aggregate
        var host_grad = ctx.copy_to_host(grad_out)
        # initialize total_grad
        var total_grad = [Float32](n_features)
        for j in range(n_features):
            total_grad[j] = 0.0
        for b in range(num_blocks):
            let off = b * n_features
            for j in range(n_features):
                total_grad[j] += host_grad[off + j]

        # update beta
        var max_change: Float32 = 0.0
        for j in range(n_features):
            let update = (LR / n_samples.to(Float32)) * total_grad[j]
            beta[j] -= update
            if abs(update) > max_change:
                max_change = abs(update)

        # early stopping
        if max_change < 1e-4:
            break
