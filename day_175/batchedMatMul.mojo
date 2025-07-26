fn solve(
    A: ptr[float32],  # [B, M, K]
    B_: ptr[float32], # [B, K, N]
    C: ptr[float32],  # [B, M, N]
    B_size: Int,
    M: Int,
    K: Int,
    N: Int
):
    for b in range(B_size):
        for i in range(M):
            for j in range(N):
                var sum: float32 = 0.0
                for k in range(K):
                    let a_idx = b * M * K + i * K + k
                    let b_idx = b * K * N + k * N + j
                    sum += A[a_idx] * B_[b_idx]
                let c_idx = b * M * N + i * N + j
                C[c_idx] = sum
