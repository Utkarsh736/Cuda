//!POPCORN leaderboard amd-identity

#include <hip/hip_runtime.h>

// Identity kernel: each thread copies one element from input to output
extern "C" __global__ void identity_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           size_t N) {
    size_t idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;  // Thread-global index
    if (idx < N) {
        output[idx] = input[idx];  // Copy value
    }
}

// Host wrapper called by the evaluation harness
void custom_kernel(const torch::Tensor& input_tensor,
                   torch::Tensor& output_tensor) {
    const float* input = input_tensor.data_ptr<float>();
    float* output      = output_tensor.data_ptr<float>();
    size_t N           = input_tensor.numel();

    // Configure launch: 256 threads per block
    size_t block_size = 256;
    size_t num_blocks = (N + block_size - 1) / block_size;

    // Launch HIP kernel 0
    hipLaunchKernelGGL(identity_kernel,
                       dim3(num_blocks), dim3(block_size),
                       0, 0,
                       input, output, N);

    // Ensure completion before returning 1
    hipDeviceSynchronize();
}