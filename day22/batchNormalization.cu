#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256  // Number of threads per block
#define EPSILON 1e-5f   // Small constant for numerical stability

// CUDA Kernel for Batch Normalization
__global__ void batch_norm_kernel(float *input, float *output, float *gamma, float *beta, float *mean, float *variance, int batch_size, int feature_dim) {
    int feature_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (feature_idx >= feature_dim) return;

    // Compute mean and variance for this feature across all samples
    float sum = 0.0f, sum_sq = 0.0f;

    for (int i = 0; i < batch_size; i++) {
        float val = input[i * feature_dim + feature_idx];
        sum += val;
        sum_sq += val * val;
    }

    float mean_val = sum / batch_size;
    float var_val = (sum_sq / batch_size) - (mean_val * mean_val);

    // Store mean and variance for later use
    mean[feature_idx] = mean_val;
    variance[feature_idx] = var_val;

    // Normalize and apply scale/shift
    for (int i = 0; i < batch_size; i++) {
        int index = i * feature_dim + feature_idx;
        float norm_x = (input[index] - mean_val) / sqrtf(var_val + EPSILON);
        output[index] = gamma[feature_idx] * norm_x + beta[feature_idx];
    }
}

// Host function to launch CUDA kernel
void batch_norm(float *h_input, float *h_output, float *h_gamma, float *h_beta, int batch_size, int feature_dim) {
    float *d_input, *d_output, *d_gamma, *d_beta, *d_mean, *d_variance;

    size_t bytes = batch_size * feature_dim * sizeof(float);
    size_t param_bytes = feature_dim * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_gamma, param_bytes);
    cudaMalloc(&d_beta, param_bytes);
    cudaMalloc(&d_mean, param_bytes);
    cudaMalloc(&d_variance, param_bytes);

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_bytes, cudaMemcpyHostToDevice);

    // Launch Batch Norm kernel (one thread per feature)
    int num_blocks = (feature_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batch_norm_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, d_gamma, d_beta, d_mean, d_variance, batch_size, feature_dim);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_mean);
    cudaFree(d_variance);
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
