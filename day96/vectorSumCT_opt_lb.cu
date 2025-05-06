//!POPCORN leaderboard vectorsum

#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int THREADS = 256;      // Threads per block
constexpr int WARPS = THREADS/32; // Number of warps per block

// Phase 1: Block-level partial sum with warp shuffle and shared memory
__global__ void block_reduce_kernel(const float* __restrict__ x,
                                    float* __restrict__ block_sums,
                                    int N) {
    // Grid-stride index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Partial sum in register
    float sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        sum += x[i];
    }

    // Intra-warp reduction using shuffle-down
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);  // warp shuffle 1
    }

    // Warp leader writes to shared memory
    __shared__ float smem[WARPS];
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x % 32;
    if (lane == 0) {
        smem[warp_id] = sum;                              // write per-warp sum 2
    }
    __syncthreads();

    // First warp aggregates warp sums
    if (warp_id == 0) {
        sum = (lane < WARPS) ? smem[lane] : 0.0f;
        for (int offset = WARPS/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) block_sums[blockIdx.x] = sum;      // store block sum 3
    }
}

// Phase 2: Final reduction of block sums on host
torch::Tensor custom_kernel(torch::Tensor input) {
    input = input.contiguous().cuda();
    int N = input.numel();
    int blocks = (N + THREADS - 1) / THREADS;

    // Allocate per-block sums
    auto block_sums = torch::empty({blocks}, input.options());
    // Launch block reduction
    block_reduce_kernel<<<blocks, THREADS>>>(
        input.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        N
    );
    cudaDeviceSynchronize();  // ensure completion 4

    // Final sum on host (or with small kernel)
    float result = block_sums.sum().item<float>();         // host-side reduction 5
    return torch::tensor(result, input.options());
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_kernel", &custom_kernel, "Optimized Vector Sum Reduction (CUDA)");
}