#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024  // Optimal for most GPUs

// **Kernel 1: Block-Level Sum Reduction using Shared Memory**
__global__ void sum_reduction_shared(int *arr, int *partial_sums, int n) {
    __shared__ int s_data[THREADS_PER_BLOCK];  // Shared memory for partial sums
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load data into shared memory, handle boundary condition
    if (idx < n)
        s_data[tid] = arr[idx] + (idx + blockDim.x < n ? arr[idx + blockDim.x] : 0);
    else
        s_data[tid] = 0;
    __syncthreads();

    // Perform sum reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // Store block-wise partial sum in global memory
    if (tid == 0) partial_sums[blockIdx.x] = s_data[0];
}

// **Kernel 2: Final Reduction on Partial Sums**
__global__ void final_sum(int *partial_sums, int *result, int n) {
    __shared__ int s_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    if (tid < n) s_data[tid] = partial_sums[tid];
    else s_data[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) *result = s_data[0];
}

int main() {
    int n = 1 << 24;  // 16M elements
    size_t bytes = n * sizeof(int);

    // Allocate and initialize host memory
    int *h_array = new int[n];
    for (int i = 0; i < n; i++) h_array[i] = 1;  // Initialize array with 1s for testing

    // Allocate device memory
    int *d_array, *d_partial_sums, *d_result;
    cudaMalloc(&d_array, bytes);
    cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

    // Compute number of blocks for first reduction
    int num_blocks = (n + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    cudaMalloc(&d_partial_sums, num_blocks * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Launch kernel for block-level reduction
    sum_reduction_shared<<<num_blocks, THREADS_PER_BLOCK>>>(d_array, d_partial_sums, n);

    // Launch kernel for final sum reduction
    final_sum<<<1, THREADS_PER_BLOCK>>>(d_partial_sums, d_result, num_blocks);

    // Copy result back to host
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Final Sum: " << h_result << std::endl;

    // Cleanup
    delete[] h_array;
    cudaFree(d_array);
    cudaFree(d_partial_sums);
    cudaFree(d_result);

    return 0;
}
