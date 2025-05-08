#!POPCORN leaderboard amd-mixture-of-experts

import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def moe_kernel(
    x_ptr,            # [T, D] input tokens
    router_w_ptr,     # [D, E] router weights
    expert1_w_ptr,    # [E, D, F] expert first-layer weights
    expert2_w_ptr,    # [E, F, D] expert second-layer weights
    out_ptr,          # [T, D] output
    prob_ptr,         # [T, K] topk routing probs
    idx_ptr,          # [T, K] topk expert indices
    T, D, E, F, K,
    BLOCK:T.input,     # number of tokens per block
    D_BLOCK:tl.constexpr,  
    F_BLOCK:tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < T

    # Load token block [BLOCK, D]
    x = tl.load(x_ptr + offs[:, None]*D + tl.arange(0, D)[None, :], mask=mask[:, None])

    # Load top-k indices and probs for this block
    idx = tl.load(idx_ptr + offs[:, None]*K + tl.arange(0, K)[None, :], mask=mask[:, None])   # [BLOCK,K]
    prob = tl.load(prob_ptr + offs[:, None]*K + tl.arange(0, K)[None, :], mask=mask[:, None])  # [BLOCK,K]

    # Accumulator
    out = tl.zeros((BLOCK, D), dtype=tl.float32)

    # For each selected expert slot
    for i in range(K):
        # Gather which expert e to use
        e = idx[:, i].to(tl.int32)  # [BLOCK]
        w = prob[:, i][:, None]     # [BLOCK,1]

        # First‐layer: h = x @ W1_e^T + b1_e
        # We tile over D_BLOCK and F_BLOCK
        h = tl.zeros((BLOCK, F), dtype=tl.float32)
        for d0 in range(0, D, D_BLOCK):
            a = x[:, d0:d0+D_BLOCK]  # [BLOCK,D_BLOCK]
            # Load W1_e[d0:d0+D_BLOCK, :]
            w1 = tl.load(expert1_w_ptr + e[:, None]*D*F + (d0+tl.arange(0,D_BLOCK)[None,:])*F + tl.arange(0,F)[None,None,:])
            h += tl.dot(a, w1)  # [BLOCK,F]
        h = tl.relu(h)

        # Second‐layer: y = h @ W2_e^T + b2_e
        y = tl.zeros((BLOCK, D), dtype=tl.float32)
        for f0 in range(0, F, F_BLOCK):
            b = h[:, f0:f0+F_BLOCK]  # [BLOCK,F_BLOCK]
            w2 = tl.load(expert2_w_ptr + e[:, None]*F*D + (f0+tl.arange(0,F_BLOCK)[None,None,:])*D + tl.arange(0,D)[None,None,None,:])
            y += tl.dot(b, w2)  # [BLOCK,D]
        
        # Accumulate weighted by routing prob
        out += w * y

    # Store output
    tl.store(out_ptr + offs[:, None]*D + tl.arange(0, D)[None, :], out, mask=mask[:, None])

def custom_kernel(data: input_t) -> output_t:
    """
    Triton‐based DeepSeek‐style MoE layer.
    """
    x, weights, config = data
    bs, seq_len, D = x.shape
    E = config["num_experts"]
    F = config["d_ff"]
    K = config["top_k"]
    T = bs * seq_len

    # Flatten tokens
    x_flat = x.contiguous().view(T, D).cuda()
    # Router weights
    W_router = weights["router_w"].cuda()        # [D,E]
    # Expert weights
    W1 = weights["expert_w1"].cuda()             # [E,D,F]
    W2 = weights["expert_w2"].cuda()             # [E,F,D]

    # 1) Compute router logits + softmax
    logits = x_flat @ W_router                     # [T,E]
    probs = torch.softmax(logits, dim=-1)          # [T,E]

    # 2) Top-k routing
    topk_vals, topk_idx = torch.topk(probs, K, dim=-1)  # both [T,K]

    # 3) Prepare output buffer
    out = torch.zeros_like(x_flat, device=x.device)

    # 4) Launch Triton MoE kernel
    BLOCK = 256
    D_BLOCK = 64
    F_BLOCK = 64
    grid = ( (T + BLOCK -1)//BLOCK, )
    moe_kernel[grid](
        x_flat.data_ptr(), W_router.data_ptr(),
        W1.data_ptr(), W2.data_ptr(),
        out.data_ptr(), topk_vals.data_ptr(), topk_idx.data_ptr(),
        T, D, E, F, K,
        BLOCK, D_BLOCK, F_BLOCK
    )

    # 5) Compute auxiliary load-balancing loss
    load = (probs.mean(0) - 1.0/E).pow(2).sum() * E

    # Reshape output
    output = out.view(bs, seq_len, D)
    aux = {"aux_loss": load, "router_probs": probs.view(bs, seq_len, E)}
    return output, aux