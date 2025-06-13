// dot_product.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void dot_kernel(const float* A, const float* B, float* partial_sum, int N) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_idx = threadIdx.x;

    float temp = 0.0f;
    while (tid < N) {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cache_idx] = temp;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cache_idx < s)
            cache[cache_idx] += cache[cache_idx + s];
        __syncthreads();
    }

    // Write result of this block to global memory
    if (cache_idx == 0)
        partial_sum[blockIdx.x] = cache[0];
}

// Host solve function
extern "C" void solve(const float* A, const float* B, int N, float* output) {
    float *d_A, *d_B, *d_partial;
    int threads = 256;
    int blocks = 256;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_partial, blocks * sizeof(float));

    dot_kernel<<<blocks, threads>>>(d_A, d_B, d_partial, N);
    cudaDeviceSynchronize();

    float h_partial[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (int i = 0; i < blocks; ++i) result += h_partial[i];

    *output = result;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);
}

// Optional test main
#ifdef TEST_DOT
int main() {
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    int N = sizeof(A) / sizeof(A[0]);
    float result = 0.0f;

    solve(A, B, N, &result);
    printf("Dot Product: %.1f\n", result);
    return 0;
}
#endif
