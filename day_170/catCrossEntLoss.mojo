from gpu.id import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from math import exp, log
from memory import UnsafePointer, atomic_add

const BLOCK_SIZE = 256

fn cross_entropy_kernel(
    logits: UnsafePointer[Float32],
    true_labels: UnsafePointer[Int32],
    N: Int32,
    C: Int32,
    partial_loss: UnsafePointer[Float32]
):
    idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= N:
        return

    var max_logit: Float32 = logits[idx * C]
    for j in range(1, C):
        val = logits[idx * C + j]
        if val > max_logit:
            max_logit = val

    var sum_exp: Float32 = 0.0
    for j in range(C):
        sum_exp += exp(logits[idx * C + j] - max_logit)

    log_sum_exp = log(sum_exp)
    true_class = true_labels[idx]
    logit_true = logits[idx * C + true_class]
    loss_val = - (logit_true - max_logit - log_sum_exp)

    atomic_add(partial_loss, loss_val)

@export
def solve(
    logits: UnsafePointer[Float32],
    true_labels: UnsafePointer[Int32],
    N: Int32,
    C: Int32,
    loss: UnsafePointer[Float32]
):
    var ctx = DeviceContext()
    var d_partial_loss = ctx.alloc_device_memory 
    ctx.memset(d_partial_loss, 0, 1)

    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    ctx.enqueue_function[cross_entropy_kernel](
        logits, true_labels, N, C, d_partial_loss,
        grid_dim = (grid_size,),
        block_dim = (BLOCK_SIZE,)
    )
    ctx.copy_to_host(d_partial_loss, loss, 1)
    ctx.synchronize()
    
    # Average the loss
    loss[0] = loss[0] / float32(N)
