// prefix_sum.cu
#include <cuda_runtime.h>
#include <cstdio>

// Upsweep phase: build sum tree
__global__ void upsweep(float* data, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (i + 1) * stride * 2 - 1;
    if (idx < N)
        data[idx] += data[idx - stride];
}

// Downsweep phase: propagate prefix
__global__ void downsweep(float* data, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (i + 1) * stride * 2 - 1;
    if (idx < N) {
        float t = data[idx - stride];
        data[idx - stride] = data[idx];
        data[idx] += t;
    }
}

// Host solve function: prefix sum of input[0..N-1] -> output[0..N-1]
extern "C" void solve(const float* input, float* output, int N) {
    int size = 1;
    while (size < N) size <<= 1; // next power of 2
    float* d_data;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemcpy(d_data, input, N * sizeof(float), cudaMemcpyHostToDevice);
    if (size > N)
        cudaMemset(d_data + N, 0, (size - N) * sizeof(float)); // pad with 0s

    int threads = 256;
    for (int stride = 1; stride < size; stride *= 2) {
        int blocks = (size / (2 * stride) + threads - 1) / threads;
        upsweep<<<blocks, threads>>>(d_data, size, stride);
        cudaDeviceSynchronize();
    }

    // Set last element to 0 before downsweep
    cudaMemset(d_data + size - 1, 0, sizeof(float));

    for (int stride = size / 2; stride >= 1; stride /= 2) {
        int blocks = (size / (2 * stride) + threads - 1) / threads;
        downsweep<<<blocks, threads>>>(d_data, size, stride);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(output, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// Optional test
#ifdef TEST_PREFIX
int main() {
    float in[] = {5.0f, -2.0f, 3.0f, 1.0f, -4.0f};
    int N = sizeof(in) / sizeof(in[0]);
    float out[N];

    solve(in, out, N);

    printf("Prefix Sum:\n");
    for (int i = 0; i < N; i++) printf("%.1f ", out[i]);
    printf("\n");
    return 0;
}
#endif
