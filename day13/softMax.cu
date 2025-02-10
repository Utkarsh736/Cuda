#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define N 1024  // Number of elements
#define THREADS_PER_BLOCK 256

// **Kernel to compute exponentials and sum**
__global__ void exp_sum_kernel(float *d_input, float *d_exp, float *d_sum) {
    __shared__ float shared_sum[THREADS_PER_BLOCK];  // Shared memory for partial sums
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute exponentials
    float val = expf(d_input[tid]);
    d_exp[tid] = val;
    shared_sum[threadIdx.x] = val;
    __syncthreads();

    // **Reduction to compute sum**
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Store block sum in global memory
    if (threadIdx.x == 0) atomicAdd(d_sum, shared_sum[0]);
}

// **Kernel to normalize exponentials (compute softmax)**
__global__ void softmax_kernel(float *d_exp, float *d_output, float *d_sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    d_output[tid] = d_exp[tid] / *d_sum;  // Normalize each value
}

// **Host function to launch softmax**
void softmax(float *h_input, float *h_output) {
    float *d_input, *d_exp, *d_output, *d_sum;
    size_t bytes = N * sizeof(float);
    
    // Allocate memory on device
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_exp, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));  // Initialize sum to 0

    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Grid and block configuration
    int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // **Launch kernels**
    exp_sum_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_exp, d_sum);
    cudaDeviceSynchronize();
    softmax_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_exp, d_output, d_sum);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_exp);
    cudaFree(d_output);
    cudaFree(d_sum);
}

int main() {
    float h_input[N], h_output[N];

    // Initialize input array with random values
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;  // Random float in [0,1]
    }

    // Compute softmax
    softmax(h_input, h_output);

    // Print first 10 results
    std::cout << "Softmax Output (First 10 elements):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
