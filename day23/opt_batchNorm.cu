#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256  // Number of threads per block
#define EPSILON 1e-5f   // Small constant for numerical stability

// CUDA Kernel: Parallel reduction for mean and variance in shared memory
__global__ void batch_norm_kernel(float *input, float *output, float *gamma, float *beta, int batch_size, int feature_dim) {
    __shared__ float mean_shared[BLOCK_SIZE];
    __shared__ float variance_shared[BLOCK_SIZE];

    int feature_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (feature_idx >= feature_dim) return;

    // Compute mean using parallel reduction
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        float val = input[i * feature_dim + feature_idx];
        sum += val;
        sum_sq += val * val;
    }

    // Store partial sums in shared memory
    mean_shared[threadIdx.x] = sum;
    variance_shared[threadIdx.x] = sum_sq;
    __syncthreads();

    // Reduce mean and variance in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            mean_shared[threadIdx.x] += mean_shared[threadIdx.x + stride];
            variance_shared[threadIdx.x] += variance_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Compute final mean and variance
    float mean_val = mean_shared[0] / batch_size;
    float var_val = (variance_shared[0] / batch_size) - (mean_val * mean_val);

    // Normalize and apply scale/shift
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        int index = i * feature_dim + feature_idx;
        float norm_x = (input[index] - mean_val) / sqrtf(var_val + EPSILON);
        output[index] = gamma[feature_idx] * norm_x + beta[feature_idx];
    }
}

// Host function to launch CUDA kernel
void batch_norm(float *h_input, float *h_output, float *h_gamma, float *h_beta, int batch_size, int feature_dim) {
    float *d_input, *d_output, *d_gamma, *d_beta;

    size_t bytes = batch_size * feature_dim * sizeof(float);
    size_t param_bytes = feature_dim * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_gamma, param_bytes);
    cudaMalloc(&d_beta, param_bytes);

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_bytes, cudaMemcpyHostToDevice);

    // Launch optimized Batch Norm kernel
    int num_blocks = (feature_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batch_norm_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, d_gamma, d_beta, batch_size, feature_dim);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
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
    const int batch_size = 4;
    const int feature_dim = 3;

    float h_input[batch_size * feature_dim] = {
        1.0, 2.0, 3.0, 
        4.0, 5.0, 6.0, 
        7.0, 8.0, 9.0, 
        10.0, 11.0, 12.0
    };
    float h_output[batch_size * feature_dim];

    float h_gamma[feature_dim] = {1.0, 1.0, 1.0};  // Scale
    float h_beta[feature_dim] = {0.0, 0.0, 0.0};   // Shift

    std::cout << "Input:\n";
    printArray(h_input, batch_size * feature_dim);

    // Compute Batch Normalization
    batch_norm(h_input, h_output, h_gamma, h_beta, batch_size, feature_dim);

    std::cout << "Batch Norm Output:\n";
    printArray(h_output, batch_size * feature_dim);

    return 0;
}
