import torch, triton, triton.language as tl

@triton.jit
def moe_dispatch_kernel(
    x_ptr,           # [T, d_hidden]
    router_w_ptr,    # [d_hidden, E]
    expert_w1_ptr,   # [E, d_hidden, d_ff]
    expert_w2_ptr,   # [E, d_ff, d_hidden]
    out_ptr,         # [T, d_hidden]
    T, d_hidden, E, d_ff, k,
):
    tid = tl.program_id(0)  # each program handles token t
    if tid >= T: return

    # 1. Load x[t]
    x = tl.load(x_ptr + tid*d_hidden + tl.arange(0, d_hidden))

    # 2. Compute router logits [E]
    logits = tl.dot(x, router_w_ptr)    # tl.dot over hidden dim

    # 3. Softmax (block‐local) and top‐k (approximate)
    probs = tl.softmax(logits)
    topk_idx, topk_val = tl.topk(probs, k=k)

    # 4. For each selected expert
    out = tl.zeros([d_hidden], tl.float32)
    for i in range(k):
        e = topk_idx[i]   # expert index
        w = topk_val[i]   # weight

        # MLP expert e: x→h→out_e
        h = tl.relu( tl.dot(x, expert_w1_ptr + e*d_hidden*d_ff) )
        out_e = tl.dot(h, expert_w2_ptr + e*d_ff*d_hidden)

        out += w * out_e

    # 5. Store output
    tl.store(out_ptr + tid*d_hidden + tl.arange(0, d_hidden), out)

def custom_kernel(data):
    input, weights, config = data
    bs, seq_len, d = input.shape
    T = bs * seq_len
    E = config['num_experts']
    d_ff = config['d_ff']
    k = config['top_k']

    # Prepare tensors & pointers...
    x = input.contiguous().cuda().view(T, d)
    out = torch.empty_like(x)
    # Pack weight ptrs...
    grid = (T,)

    moe_dispatch_kernel[grid](
        x.data_ptr(), weights['router_w'].data_ptr(),
        weights['expert_w1'].data_ptr(), weights['expert_w2'].data_ptr(),
        out.data_ptr(), T, d, E, d_ff, k
    )
    output = out.view(bs, seq_len, d)
    # Compute aux losses in Python
    router_probs = torch.softmax(x @ weights['router_w'], dim=-1)
    aux_loss = E * ((router_probs.mean(0) - 1/E)**2).sum()
    return output, {'aux_loss': aux_loss, 'router_probs': router_probs.view(bs, seq_len, E)}