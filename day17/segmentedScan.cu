#include <iostream>
#include <cuda.h>

#define BLOCK_SIZE 256  // Number of threads per block

// CUDA kernel for segmented scan
__global__ void segmented_scan_kernel(int *d_input, int *d_flags, int *d_output, int n) {
    __shared__ int temp[BLOCK_SIZE];
    __shared__ int flag_temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        temp[tid] = d_input[gid];      // Load input to shared memory
        flag_temp[tid] = d_flags[gid]; // Load flags to shared memory
    } else {
        temp[tid] = 0;
        flag_temp[tid] = 0;
    }
    __syncthreads();

    // Hillis-Steele scan with segment awareness
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        if (tid >= offset && flag_temp[tid] == 0) {
            val = temp[tid - offset];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write results to global memory
    if (gid < n) {
        d_output[gid] = temp[tid];
    }
}

// Host function to run the segmented scan
void segmented_scan(int *h_input, int *h_flags, int *h_output, int n) {
    int *d_input, *d_flags, *d_output;
    
    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, n * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel execution
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    segmented_scan_kernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_flags, d_output, n);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_flags);
    cudaFree(d_output);
}

// Helper function to print an array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Main function
int main() {
    const int n = 10;
    int h_input[n]  = {1, 2, 3, 4, 5,  1, 1, 1, 1, 1};
    int h_flags[n]  = {1, 0, 0, 0, 0,  1, 0, 0, 0, 0};  // 2 segments
    int h_output[n];

    std::cout << "Input:  ";
    printArray(h_input, n);
    std::cout << "Flags:  ";
    printArray(h_flags, n);

    // Perform segmented scan
    segmented_scan(h_input, h_flags, h_output, n);

    std::cout << "Output: ";
    printArray(h_output, n);

    return 0;
}
