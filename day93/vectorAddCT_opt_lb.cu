//!POPCORN leaderboard vectoradd

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int THREADS = 256;

// CUDA kernel: FP16 vector add with half2 intrinsics and grid-stride loop
__global__ void vectoradd_half2(const __half2* __restrict__ A,
                                const __half2* __restrict__ B,
                                __half2* __restrict__ C,
                                int total_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; idx < total_pairs; idx += stride) {
        // Use half2 add intrinsic for two FP16 elements at once 0
        __half2 a = A[idx];
        __half2 b = B[idx];
        C[idx] = __hadd2(a, b);
    }
}

// Host binding for PyTorch extension
torch::Tensor custom_kernel(torch::Tensor A, torch::Tensor B) {
    // Input tensors are (N,N) FP16
    int64_t N = A.size(0);
    int64_t total = N * N;
    // Number of half2 pairs (ceil)
    int total_pairs = (total + 1) / 2;                             
    // Allocate output tensor
    auto C = torch::empty({N, N}, A.options());
    // Cast pointers to __half2*
    const __half2* A2 = reinterpret_cast<const __half2*>(A.data_ptr<at::Half>()); 
    const __half2* B2 = reinterpret_cast<const __half2*>(B.data_ptr<at::Half>()); 
    __half2*       C2 = reinterpret_cast<__half2*>(C.data_ptr<at::Half>());       

    // Launch kernel: grid-stride for large N 1
    int blocks = (total_pairs + THREADS - 1) / THREADS;
    vectoradd_half2<<<blocks, THREADS>>>(A2, B2, C2, total_pairs);
    cudaDeviceSynchronize();                                         // ensure completion 2
    return C;
}

// PyTorch module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_kernel", &custom_kernel, "FP16 Vector Add (CUDA)");
}