fn clamp_to_int8(val: Int32) -> Int8:
    if val < -128:
        return -128
    elif val > 127:
        return 127
    else:
        return val.to_int8()

fn solve(
    A: ptr[Int8],  # [M × K]
    B: ptr[Int8],  # [K × N]
    C: ptr[Int8],  # [M × N]
    M: Int,
    N: Int,
    K: Int,
    scale_A: Float32,
    scale_B: Float32,
    scale_C: Float32,
    zero_point_A: Int32,
    zero_point_B: Int32,
    zero_point_C: Int32
):
    let scale = (scale_A * scale_B) / scale_C

    for i in range(M):
        for j in range(N):
            var acc: Int32 = 0
            for k in range(K):
                let a_idx = i * K + k
                let b_idx = k * N + j
                let a_val: Int32 = A[a_idx].to_int32() - zero_point_A
                let b_val: Int32 = B[b_idx].to_int32() - zero_point_B
                acc += a_val * b_val

            # Apply scaling and zero point
            var scaled: Float32 = acc.to_float32() * scale
            var rounded: Int32 = round(scaled).to_int32() + zero_point_C
            C[i * N + j] = clamp_to_int8(rounded)
