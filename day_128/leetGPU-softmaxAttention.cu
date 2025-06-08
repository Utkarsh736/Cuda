#include <cuda_runtime.h>
#include <math.h>

:contentReference[oaicite:7]{index=7}
    :contentReference[oaicite:8]{index=8}
    :contentReference[oaicite:9]{index=9}
    :contentReference[oaicite:10]{index=10}
    :contentReference[oaicite:11]{index=11}
    :contentReference[oaicite:12]{index=12}
{
    :contentReference[oaicite:13]{index=13}
    :contentReference[oaicite:14]{index=14}
    :contentReference[oaicite:15]{index=15} 
    :contentReference[oaicite:16]{index=16} 

    int q_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (q_row >= M) return;

    float out_val = 0.0f;
    const float inv_sqrt_d = rsqrtf((float)d);

    for (int k_base = 0; k_base < N; k_base += blockDim.x) {
        int Kn = min(blockDim.x, N - k_base);
        
        // Load K and V tiles into shared memory
        if (threadIdx.x < Kn && threadIdx.y < d) {
            Ks[threadIdx.x * d + threadIdx.y] = 
                K[(k_base + threadIdx.x) * d + threadIdx.y];
            Vs[threadIdx.x * d + threadIdx.y] = 
                V[(k_base + threadIdx.x) * d + threadIdx.y];
        }
        // Load Q for this query row
        if (threadIdx.x == 0 && threadIdx.y < d && q_row < M) {
            Qs[threadIdx.y] = Q[q_row * d + threadIdx.y];
        }
        __syncthreads();

        // Compute attention: dot(Qs, each row of Ks)
        float max_score = -FLT_MAX;
        float sum_exp = 0.0f;
        for (int j = 0; j < Kn; ++j) {
            float dot = 0;
            for (int t = 0; t < d; ++t)
                dot += Qs[t] * Ks[j * d + t];
            float score = dot * inv_sqrt_d;
            if (score > max_score) max_score = score;
        }
        // Softmax
        for (int j = 0; j < Kn; ++j) {
            float dot = 0;
            for (int t = 0; t < d; ++t)
                dot += Qs[t] * Ks[j * d + t];
            float score = expf(dot*inv_sqrt_d - max_score);
            sum_exp += score;
            // accumulate weighted V
            for (int t = 0; t < d; ++t)
                out_val += (score * Vs[j * d + t]);
        }
        __syncthreads();
    }

    // Normalize and store result
    if (q_row < M) {
        float* out = output + q_row * d;
        for (int t = 0; t < d; ++t)
            out[t] = out_val / sum_exp;
    }
}

:contentReference[oaicite:17]{index=17}
           :contentReference[oaicite:18]{index=18}
{
    :contentReference[oaicite:19]{index=19}
    :contentReference[oaicite:20]{index=20}
              :contentReference[oaicite:21]{index=21}

    size_t smem = d * threadsPerBlock.y * sizeof(float)
                   + 2 * d * threadsPerBlock.x * sizeof(float);

    cudaMalloc(/*... allocate device Q,K,V,output ...*/);
    // copy inputs to device

    softmaxAttentionKernel<<<grid, threadsPerBlock, smem>>>(
         d_Q, d_K, d_V, d_output, M, N, d);

    // copy output back and free memory
}
