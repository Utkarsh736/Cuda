#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define MASK_LENGTH 5  

// Constant memory for the stencil mask
__constant__ int d_mask[MASK_LENGTH];

// CUDA kernel
__global__ void stencil_1d(int *input, int *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int r = MASK_LENGTH / 2; // Mask radius
    int temp = 0;

    // Boundary Check
    if (tid < n) {
        for (int j = -r; j <= r; j++) {
            int index = tid + j;
            if (index >= 0 && index < n) {
                temp += input[index] * d_mask[j + r]; // Apply stencil
            }
        }
        output[tid] = temp;
    }
}

// Verification Function
void verify_result(int *input, int *mask, int *output, int n) {
    int r = MASK_LENGTH / 2;
    for (int i = 0; i < n; i++) {
        int temp = 0;
        for (int j = -r; j <= r; j++) {
            int index = i + j;
            if (index >= 0 && index < n) {
                temp += input[index] * mask[j + r];
            }
        }
        assert(temp == output[i]);
    }
}

int main() {
    int n = 1 << 20;  // Number of elements
    int bytes = n * sizeof(int);
    int mask_bytes = MASK_LENGTH * sizeof(int);

    // Allocate host memory
    int *h_input = (int *)malloc(bytes);
    int *h_output = (int *)malloc(bytes);
    int *h_mask = (int *)malloc(mask_bytes);

    // Initialize input and mask
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() % 100;
    }
    for (int i = 0; i < MASK_LENGTH; i++) {
        h_mask[i] = rand() % 10;
    }

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, mask_bytes);

    // Configure CUDA kernel
    int THREADS = 256;
    int BLOCKS = (n + THREADS - 1) / THREADS;

    // Launch kernel
    stencil_1d<<<BLOCKS, THREADS>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    verify_result(h_input, h_mask, h_output, n);

    printf("COMPLETED SUCCESSFULLY\n");

    // Free memory
    free(h_input);
    free(h_output);
    free(h_mask);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
