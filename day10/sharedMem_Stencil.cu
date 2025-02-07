#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define MASK_LENGTH 5  // Stencil size
#define RADIUS (MASK_LENGTH / 2)  // Half the mask size
#define THREADS_PER_BLOCK 256  // Number of threads per block

// Constant memory for the stencil mask
__constant__ int d_mask[MASK_LENGTH];

// CUDA kernel using shared memory for stencil computation
__global__ void stencil_1d_shared(int *input, int *output, int n) {
    // Shared memory allocation
    __shared__ int s_data[THREADS_PER_BLOCK + 2 * RADIUS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load main data into shared memory
    if (gid < n) {
        s_data[RADIUS + tid] = input[gid];
    }

    // Load left halo elements
    if (tid < RADIUS) {
        int left_index = gid - RADIUS;
        s_data[tid] = (left_index >= 0) ? input[left_index] : 0;
    }

    // Load right halo elements
    if (tid >= blockDim.x - RADIUS) {
        int right_index = gid + RADIUS;
        s_data[RADIUS + tid + RADIUS] = (right_index < n) ? input[right_index] : 0;
    }

    __syncthreads();

    // Compute stencil if within valid range
    if (gid < n) {
        int temp = 0;
        for (int j = 0; j < MASK_LENGTH; j++) {
            temp += s_data[tid + j] * d_mask[j];
        }
        output[gid] = temp;
    }
}

// Verification Function
void verify_result(int *input, int *mask, int *output, int n) {
    for (int i = 0; i < n; i++) {
        int temp = 0;
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int index = i + j;
            if (index >= 0 && index < n) {
                temp += input[index] * mask[j + RADIUS];
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

    int THREADS = THREADS_PER_BLOCK;
    int BLOCKS = (n + THREADS - 1) / THREADS;

    // Kernel Launch
    size_t shared_mem_size = (THREADS + 2 * RADIUS) * sizeof(int);
    stencil_1d_shared<<<BLOCKS, THREADS, shared_mem_size>>>(d_input, d_output, n);
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
