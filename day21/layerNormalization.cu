#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256  // Number of threads per block
#define EPSILON 1e-5f   // Small constant for numerical stability

// CUDA Kernel for Layer Normalization
__global__ void layer_norm_kernel(float *input, float *output, float *gamma, float *beta, int N, int feature_dim) {
    int batch_idx = blockIdx.x;  // Each block handles a single input
    int thread_idx = threadIdx.x;

    // Shared memory for mean and variance
    __shared__ float mean;
    __shared__ float variance;

    // Compute mean
    float sum = 0.0f;
    for (int i = thread_idx; i < feature_dim; i += blockDim.x) {
        sum += input[batch_idx * feature_dim + i];
    }

    sum /= feature_dim; // Normalize sum

    if (thread_idx == 0) mean = sum; // Store mean in shared memory
    __syncthreads();

    // Compute variance
    float sum_sq = 0.0f;
    for (int i = thread_idx; i < feature_dim; i += blockDim.x) {
        float diff = input[batch_idx * feature_dim + i] - mean;
        sum_sq += diff * diff;
    }

    sum_sq /= feature_dim; // Normalize variance

    if (thread_idx == 0) variance = sum_sq;
    __syncthreads();

    // Normalize and apply scale and shift
    for (int i = thread_idx; i < feature_dim; i += blockDim.x) {
        float norm_x = (input[batch_idx * feature_dim + i] - mean) / sqrtf(variance + EPSILON);
        output[batch_idx * feature_dim + i] = norm_x * gamma[i] + beta[i];
    }
}

// Host function to launch CUDA kernel
void layer_norm(float *h_input, float *h_output, float *h_gamma, float *h_beta, int batch_size, int feature_dim) {
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

    // Launch Layer Norm kernel (one block per input)
    layer_norm_kernel<<<batch_size, BLOCK_SIZE>>>(d_input, d_output, d_gamma, d_beta, batch_size, feature_dim);

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
    const int batch_size = 2;
    const int feature_dim = 4;

    float h_input[batch_size * feature_dim] = {1.0, 2.0, 3.0, 4.0, 
                                               5.0, 6.0, 7.0, 8.0};
    float h_output[batch_size * feature_dim];

    float h_gamma[feature_dim] = {1.0, 1.0, 1.0, 1.0};  // Scale
    float h_beta[feature_dim] = {0.0, 0.0, 0.0, 0.0};   // Shift

    std::cout << "Input:\n";
    printArray(h_input, batch_size * feature_dim);

    // Compute Layer Normalization
    layer_norm(h_input, h_output, h_gamma, h_beta, batch_size, feature_dim);

    std::cout << "Layer Norm Output:\n";
    printArray(h_output, batch_size * feature_dim);

    return 0;
}
