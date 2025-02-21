#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define TILE_SIZE 16  // Define the tile size for shared memory

// CUDA kernel for FlashAttention
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,  // Query matrix
    const float* __restrict__ K,  // Key matrix
    const float* __restrict__ V,  // Value matrix
    float* __restrict__ output,   // Output matrix
    int seq_len,                  // Sequence length
    int head_dim)                 // Dimension of each head
{
    // Shared memory for tiles of Q, K, V
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];

    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (seq_len + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < seq_len && t * TILE_SIZE + tx < head_dim) {
            tile_Q[ty][tx] = Q[row * head_dim + t * TILE_SIZE + tx];
            tile_K[ty][tx] = K[row * head_dim + t * TILE_SIZE + tx];
            tile_V[ty][tx] = V[row * head_dim + t * TILE_SIZE + tx];
        } else {
            tile_Q[ty][tx] = 0.0f;
            tile_K[ty][tx] = 0.0f;
            tile_V[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tile_Q[ty][i] * tile_K[i][tx];
        }
        __syncthreads();
    }

    // Apply softmax normalization
    float norm_factor = 1.0f / sqrtf(static_cast<float>(head_dim));
    float attention_score = expf(sum * norm_factor);

    // Write the result to the output matrix
    if (row < seq_len && col < head_dim) {
        output[row * head_dim + col] = attention_score * tile_V[ty][tx];
    }
}

// Host function to launch the FlashAttention kernel
void flash_attention(
    const float* Q, const float* K, const float* V,
    float* output, int seq_len, int head_dim)
{
    // Define block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((head_dim + TILE_SIZE - 1) / TILE_SIZE,
                 (seq_len + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    flash_attention_kernel<<<gridDim, blockDim>>>(Q, K, V, output, seq_len, head_dim);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

int main() {
    // Example usage
    const int seq_len = 128;  // Example sequence length
    const int head_dim = 64;  // Example head dimension

    // Allocate and initialize host memory
    float* h_Q = new float[seq_len * head_dim];
    float* h_K = new float[seq_len * head_dim];
    float* h_V = new float[seq_len * head_dim];
    float* h_output = new float[seq_len * head_dim];

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_K, seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_V, seq_len * head_dim * sizeof(float));
    cudaMalloc(&d_output, seq_len * head_dim * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_Q, h_Q, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch FlashAttention
    flash_attention(d_Q, d_K, d_V, d_output, seq_len, head_dim);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);


    // Clean up
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_output;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    return 0;
}
