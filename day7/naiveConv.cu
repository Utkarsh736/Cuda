#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

// 1-D convolution kernel
__global__ void convolution_1d(int *array, int *mask, int *result, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int r = m / 2;
    int start = tid - r;
    int temp = 0;

    for (int j = 0; j < m; j++) {
        if ((start + j >= 0) && (start + j < n)) {
            temp += array[start + j] * mask[j];
        }
    }
    result[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n, int m) {
    int r = m / 2;
    for (int i = 0; i < n; i++) {
        int start = i - r;
        int temp = 0;
        for (int j = 0; j < m; j++) {
            if ((start + j >= 0) && (start + j < n)) {
                temp += array[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}

int main() {
    int n = 1 << 20; // Number of elements in result array
    int bytes_n = n * sizeof(int);
    int m = 7; // Number of elements in the convolution mask
    int bytes_m = m * sizeof(int);

    // Allocate host memory
    int *h_array = (int *)malloc(bytes_n);
    int *h_mask = (int *)malloc(bytes_m);
    int *h_result = (int *)malloc(bytes_n);

    // Initialize array and mask with random values
    for (int i = 0; i < n; i++) {
        h_array[i] = rand() % 100;
    }
    for (int i = 0; i < m; i++) {
        h_mask[i] = rand() % 10;
    }

    // Allocate device memory
    int *d_array, *d_mask, *d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_n);

    // Copy data to device
    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice);

    // Define CUDA grid and block sizes
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;

    // Launch kernel
    convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify 
    verify_result(h_array, h_mask, h_result, n, m);

    printf("COMPLETED SUCCESSFULLY\n");

    // Free memory
    free(h_array);
    free(h_mask);
    free(h_result);
    cudaFree(d_array);
    cudaFree(d_mask);
    cudaFree(d_result);

    return 0;
}
