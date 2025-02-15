#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256  // Number of threads per block

// Kernel to compute max value in a block (for numerical stability)
__global__ void find_max_kernel(float *input, float *max_vals, int n) {
    __shared__ float shared_max[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_max = -INFINITY;
    if (idx < n) {
        local_max = input[idx];
    }

    shared_max[tid] = local_max;
    __syncthreads();

    // Parallel reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_vals[blockIdx.x] = shared_max[0];
    }
}

// Kernel to compute sum of exponentials
__global__ void exp_sum_kernel(float *input, float *max_val, float *sum_vals, int n) {
    __shared__ float shared_sum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0;
    if (idx < n) {
        local_sum = expf(input[idx] - *max_val);
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_vals[blockIdx.x] = shared_sum[0];
    }
}

// Kernel to compute softmax
__global__ void softmax_kernel(float *input, float *output, float *max_val, float *sum_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx] - *max_val) / *sum_val;
    }
}

// Softmax function
void softmax(float *h_input, float *h_output, int n) {
    float *d_input, *d_output, *d_max_vals, *d_sum_vals, *d_max, *d_sum;
    
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMalloc(&d_max_vals, sizeof(float) * ((n + BLOCK_SIZE - 1) / BLOCK_SIZE));
    cudaMalloc(&d_sum_vals, sizeof(float) * ((n + BLOCK_SIZE - 1) / BLOCK_SIZE));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Step 1: Compute max for numerical stability
    find_max_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_max_vals, n);
    find_max_kernel<<<1, BLOCK_SIZE>>>(d_max_vals, d_max, grid_size);

    // Step 2: Compute sum of exponentials
    exp_sum_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_max, d_sum_vals, n);
    exp_sum_kernel<<<1, BLOCK_SIZE>>>(d_sum_vals, d_max, d_sum, grid_size);

    // Step 3: Compute softmax
    softmax_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, d_max, d_sum, n);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max_vals);
    cudaFree(d_sum_vals);
    cudaFree(d_max);
    cudaFree(d_sum);
}

// Helper function to print an array
void printArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Main function
int main() {
    const int n = 10;
    float h_input[n]  = {1.0, 2.0, 3.0, 4.0, 5.0,  6.0, 7.0, 8.0, 9.0, 10.0};
    float h_output[n];

    std::cout << "Input:  ";
    printArray(h_input, n);

    // Compute softmax
    softmax(h_input, h_output, n);

    std::cout << "Softmax Output: ";
    printArray(h_output, n);

    return 0;
}
