from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from memory import UnsafePointer
from math import ceildiv, exp

# Single-head attention kernel: processes one head per batch element
fn head_kernel(
    Qh: UnsafePointer[Float32],
    Kh: UnsafePointer[Float32],
    Vh: UnsafePointer[Float32],
    head_out: UnsafePointer[Float32],
    N: Int32,  # batch size
    L: Int32,  # sequence length
    Dh: Int32  # head-dimension = d_model/h
):
    let bid = block_idx.x
    let tid = thread_idx.x
    if bid >= N * L * Dh:
        return
    let head_idx = bid
    let batch = head_idx / (L * Dh)
    let rem = head_idx % (L * Dh)
    let pos = rem / Dh
    let dim = rem % Dh

    # compute attention for (batch, pos)
    # 1. compute dot products Qh · Khᵀ across sequence
    var max_score: Float32 = -1e30
    var scores = [Float32](L)
    for j in range(L):
        var dot: Float32 = 0.0
        for d in range(Dh):
            dot += Qh[(batch * L + pos) * Dh + d] * Kh[(batch * L + j) * Dh + d]
        scores[j] = dot / sqrt(Dh.to(Float32))
        if scores[j] > max_score:
            max_score = scores[j]
    # 2. softmax
    var sum_exp: Float32 = 0.0
    for j in range(L):
        scores[j] = exp(scores[j] - max_score)
        sum_exp += scores[j]
    for j in range(L):
        scores[j] /= sum_exp
    # 3. weighted sum into head_out
    var val: Float32 = 0.0
    for j in range(L):
        val += scores[j] * Vh[(batch * L + j) * Dh + dim]
    head_out[head_idx] = val

@export
def solve(
    Q: UnsafePointer[Float32],
    K: UnsafePointer[Float32],
    V: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int32,
    L: Int32,
    d_model: Int32,
    h: Int32
):
    let Dh = d_model / h

    # allocate per-head temporary buffers
    var ctx = DeviceContext()
    let total_heads = N * h
    let head_Q = ctx.alloc_device[Float32](total_heads * L * Dh)
    let head_K = ctx.alloc_device[Float32](total_heads * L * Dh)
    let head_V = ctx.alloc_device[Float32](total_heads * L * Dh)
    let head_out = ctx.alloc_device[Float32](total_heads * L * Dh)

    # 1. partition Q,K,V into per-head layouts
    # Single GPU kernel to split heads
    fn split_kernel(
        inp: UnsafePointer[Float32],
        out: UnsafePointer[Float32]
    ):
        let idx = block_idx.x * block_dim.x + thread_idx.x
        if idx >= N * L * d_model:
            return
        let b = idx / (L * d_model)
        let rem = idx % (L * d_model)
        let pos = rem / d_model
        let dm = rem % d_model
        let head_idx = dm / Dh
        let inner = dm % Dh
        let out_idx = ((b * h + head_idx) * L + pos) * Dh + inner
        out[out_idx] = inp[idx]

    let elems = N * L * d_model
    let blocks = ceildiv(elems, 256)
    ctx.enqueue_function[split_kernel](Q, head_Q, grid_dim=blocks, block_dim=256)
    ctx.enqueue_function[split_kernel](K, head_K, grid_dim=blocks, block_dim=256)
    ctx.enqueue_function[split_kernel](V, head_V, grid_dim=blocks, block_dim=256)
    ctx.synchronize()

    # 2. compute each head attention in parallel
    let total = total_heads * L * Dh
    let blocks2 = ceildiv(total, 256)
    ctx.enqueue_function[head_kernel](head_Q, head_K, head_V, head_out, N, L, Dh,
        grid_dim=blocks2, block_dim=256)
    ctx.synchronize()

    # 3. stitch heads back into output
    fn concat_kernel(
        inp: UnsafePointer[Float32],
        out: UnsafePointer[Float32]
    ):
        let idx = block_idx.x * block_dim.x + thread_idx.x
        if idx >= N * L * d_model:
            return
        let b = idx / (L * d_model)
        let rem = idx % (L * d_model)
        let pos = rem / d_model
        let dm = rem % d_model
        let head_idx = dm / Dh
        let inner = dm % Dh
        let in_idx = ((b * h + head_idx) * L + pos) * Dh + inner
        out[idx] = inp[in_idx]

    ctx.enqueue_function[concat_kernel](head_out, output, grid_dim=blocks, block_dim=256)
    ctx.synchronize()
