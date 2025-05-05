//!POPCORN leaderboard matmul

#include <torch/extension.h>
#include <cuda_fp16.h>

#define TILE 16

// CUDA kernel using shared memory tiling (16x16 blocks)
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K / TILE); ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    C[row * N + col] = acc;
}

// Host function: expects input tensors (A, B) as a tuple
torch::Tensor custom_matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2D matrices");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Shape mismatch");

    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, M / TILE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

// Pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_matmul", &custom_matmul, "Optimized MatMul (CUDA)");
}