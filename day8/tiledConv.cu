#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

// Length of the convolution mask
#define MASK_LENGTH 7

// Allocate space for the mask in constant memory
__constant__ int mask[MASK_LENGTH];

// 1D Convolution Kernel using Tiling and Shared Memory
__global__ void convolution_1d(int *array, int *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory allocation
    extern __shared__ int s_array[];

    int r = MASK_LENGTH / 2;   // Radius of the mask
    int d = 2 * r;             // Total padding
    int n_padded = blockDim.x + d;

    int offset = threadIdx.x + blockDim.x;
    int g_offset = blockDim.x * blockIdx.x + offset;

    // Load lower elements first
    s_array[threadIdx.x] = array[tid];

    // Load remaining elements
    if (offset < n_padded) {
        s_array[offset] = array[g_offset];
    }
    __syncthreads();

    // Convolution computation
    int temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
        temp += s_array[threadIdx.x + j] * mask[j];
    }

    result[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n) {
    for (int i = 0; i < n; i++) {
        int temp = 0;
        for (int j = 0; j < MASK_LENGTH; j++) {
            temp += array[i + j] * mask[j];
        }
        assert(temp == result[i]);
    }
}

int main() {
    int n = 1 << 20; // Number of elements in the result array
    int bytes_n = n * sizeof(int);
    size_t bytes_m = MASK_LENGTH * sizeof(int);

    int r = MASK_LENGTH / 2;
    int n_p = n + 2 * r; // Total size including padding
    size_t bytes_p = n_p * sizeof(int);

    // Allocate host memory
    int *h_array = (int *)malloc(bytes_p);
    int *h_mask = (int *)malloc(bytes_m);
    int *h_result = (int *)malloc(bytes_n);

    // Initialize the array with padding
    for (int i = 0; i < n_p; i++) {
        if ((i < r) || (i >= (n + r))) {
            h_array[i] = 0;
        } else {
            h_array[i] = rand() % 100;
        }
    }

    // Initialize the mask
    for (int i = 0; i < MASK_LENGTH; i++) {
        h_mask[i] = rand() % 10;
    }

    // Allocate device memory
    int *d_array, *d_result;
    cudaMalloc(&d_array, bytes_p);
    cudaMalloc(&d_result, bytes_n);

    // Copy data to the device
    cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

    // Copy mask to constant memory
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // CUDA kernel configuration
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;
    size_t SHMEM = (THREADS + r * 2) * sizeof(int);

    // Launch CUDA kernel
    convolution_1d<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify correctness
    verify_result(h_array, h_mask, h_result, n);

    printf("COMPLETED SUCCESSFULLY\n");

    // Free memory
    free(h_array);
    free(h_mask);
    free(h_result);
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}
