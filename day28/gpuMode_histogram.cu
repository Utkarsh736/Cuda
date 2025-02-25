//!POPCORN leaderboard histogram_cuda
#include <array>
#include <vector>
#include "task.h"
#include "utils.h"

#define BLOCK_SIZE 256  // Threads per block

// CUDA kernel for histogram computation
__global__ void histogram_kernel(const int* data, int* histogram, int size, int num_bins) {
    extern __shared__ int local_hist[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int bin;

    // Initialize shared memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Populate shared memory histogram
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        bin = data[i] / 16;  // Assign bins based on input range (0-100)
        if (bin < num_bins) {
            atomicAdd(&local_hist[bin], 1);
        }
    }
    __syncthreads();

    // Merge shared memory histogram into global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], local_hist[i]);
    }
}

// Wrapper function to call CUDA kernel
output_t custom_kernel(input_t data) {
    int size = data.size();
    int num_bins = size / 16;

    // Allocate memory on device
    int* d_data, * d_histogram;
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMalloc((void**)&d_histogram, num_bins * sizeof(int));
    cudaMemset(d_histogram, 0, num_bins * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // Configure grid and block size
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel with shared memory
    histogram_kernel<<<blocks, BLOCK_SIZE, num_bins * sizeof(int)>>>(d_data, d_histogram, size, num_bins);
    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<int> h_histogram(num_bins);
    cudaMemcpy(h_histogram.data(), d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_histogram);

    return h_histogram;
}
