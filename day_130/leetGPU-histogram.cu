// histogram.cu
#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel: compute histogram using atomic operations
__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int bin = input[idx];
        if (bin >= 0 && bin < num_bins) {
            atomicAdd(&histogram[bin], 1);
        }
    }
}

// Host solve function (matches specified signature)
extern "C"
void solve(const int* input, int* histogram, int N, int num_bins) {
    int *d_input = nullptr, *d_hist = nullptr;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_hist, num_bins * sizeof(int));

    // Copy input and initialize histogram
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, num_bins * sizeof(int));

    // Launch kernel
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    histogram_kernel<<<numBlocks, blockSize>>>(d_input, d_hist, N, num_bins);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(histogram, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_hist);
}

// Example main() for testing
#ifdef TEST_HISTOGRAM
int main() {
    const int N = 5;
    const int num_bins = 3;
    int h_input[N] = {0,1,2,1,0};
    int h_hist[num_bins] = {0};

    solve(h_input, h_hist, N, num_bins);

    printf("Histogram result:\n");
    for (int i = 0; i < num_bins; ++i) {
        printf("bin %d: %d\n", i, h_hist[i]);
    }
    return 0;
}
#endif
