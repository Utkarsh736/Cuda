#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define HISTOGRAM_BINS 256  // Number of bins for an 8-bit image
#define BLOCK_SIZE 256      // Number of threads per block

// CUDA Kernel to compute histogram using shared memory optimization
__global__ void histogram_kernel(unsigned char *d_data, int *d_histogram, int size) {
    // Shared memory histogram (private per block)
    __shared__ int local_hist[HISTOGRAM_BINS];

    // Initialize shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x < HISTOGRAM_BINS) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Populate local histogram
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        atomicAdd(&local_hist[d_data[i]], 1);
    }
    __syncthreads();

    // Merge local histograms into the global histogram
    if (threadIdx.x < HISTOGRAM_BINS) {
        atomicAdd(&d_histogram[threadIdx.x], local_hist[threadIdx.x]);
    }
}

void histogram(unsigned char *h_data, int *h_histogram, int size) {
    unsigned char *d_data;
    int *d_histogram;

    // Allocate memory on the device
    cudaMalloc((void**)&d_data, size * sizeof(unsigned char));
    cudaMalloc((void**)&d_histogram, HISTOGRAM_BINS * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, HISTOGRAM_BINS * sizeof(int));  // Initialize histogram on device

    // Define execution configuration
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogram_kernel<<<blocks, BLOCK_SIZE>>>(d_data, d_histogram, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_histogram, d_histogram, HISTOGRAM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_histogram);
}

int main() {
    const int size = 1 << 20; // 1 million elements
    unsigned char *h_data = (unsigned char*)malloc(size * sizeof(unsigned char));
    int *h_histogram = (int*)malloc(HISTOGRAM_BINS * sizeof(int));

    // Initialize input data (random 8-bit values)
    for (int i = 0; i < size; i++) {
        h_data[i] = rand() % HISTOGRAM_BINS;
    }

    // Compute histogram
    histogram(h_data, h_histogram, size);

    // Print the first 10 bins of the histogram
    for (int i = 0; i < 10; i++) {
        printf("Bin %d: %d\n", i, h_histogram[i]);
    }

    // Free host memory
    free(h_data);
    free(h_histogram);

    return 0;
}
