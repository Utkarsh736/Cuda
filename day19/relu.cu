#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Number of threads per block

// CUDA Kernel for ReLU Activation
__global__ void relu_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);  // ReLU: max(0, x)
    }
}

// Host function to launch the CUDA kernel
void relu(float *h_input, float *h_output, int n) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch ReLU kernel
    relu_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
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
    float h_input[n]  = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0};
    float h_output[n];

    std::cout << "Input:  ";
    printArray(h_input, n);

    // Compute ReLU
    relu(h_input, h_output, n);

    std::cout << "ReLU Output: ";
    printArray(h_output, n);

    return 0;
}
