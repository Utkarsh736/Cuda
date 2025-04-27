//!POPCORN leaderboard vectoradd

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Naïve half-precision vector add: one thread per element
__global__ void vector_add_half(const __half* A, const __half* B, __half* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;            // Global index 2
    int total = N * N;
    if (idx < total) {
        // Load two half values
        __half a = A[idx];                                      // Load A[idx] 3
        __half b = B[idx];                                      // Load B[idx]
        // Sum in half precision
        C[idx] = __hadd(a, b);                                  // __hadd intrinsic 4
    }
}

// Host wrapper
void custom_kernel(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    const int N = A.size(0);
    int total = N * N;

    const __half* dA = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
    const __half* dB = reinterpret_cast<const __half*>(B.data_ptr<at::Half>());
    __half*       dC = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;               // Grid size 5

    vector_add_half<<<blocks, threads>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();                                     // Ensure completion 6
}