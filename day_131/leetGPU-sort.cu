// sort.cu
#include <cuda_runtime.h>
#include <cstdio>

// Bitonic sort kernel
__global__ void bitonic_sort_step(float* data, int j, int k, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int ixj = i ^ j;
    if (ixj > i && ixj < N) {
        bool ascending = (i & k) == 0;
        if ((ascending && data[i] > data[ixj]) || (!ascending && data[i] < data[ixj])) {
            float tmp = data[i];
            data[i] = data[ixj];
            data[ixj] = tmp;
        }
    }
}

// Sorts array of size N in-place (ascending)
extern "C" void solve(float* data, int N) {
    int padded_N = 1;
    while (padded_N < N) padded_N <<= 1;

    float* d_data;
    cudaMalloc(&d_data, padded_N * sizeof(float));
    cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    if (padded_N > N) {
        float inf = __int_as_float(0x7f800000); // +INF
        cudaMemset(d_data + N, 0, (padded_N - N) * sizeof(float));
        for (int i = N; i < padded_N; ++i) cudaMemcpy(d_data + i, &inf, sizeof(float), cudaMemcpyHostToDevice);
    }

    int threads = 256;
    int blocks = (padded_N + threads - 1) / threads;

    for (int k = 2; k <= padded_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(d_data, j, k, padded_N);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// Optional test main
#ifdef TEST_SORT
int main() {
    float arr[] = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 4.0f};
    int N = sizeof(arr) / sizeof(arr[0]);

    solve(arr, N);

    printf("Sorted array:\n");
    for (int i = 0; i < N; i++) printf("%.1f ", arr[i]);
    printf("\n");
    return 0;
}
#endif
