// topk_selection.cu
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstdio>

extern "C" void solve(const float* h_input, int N, int k, float* h_output) {
    float* d_input;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Copy data back for host-based selection (for now, to keep logic simple)
    std::vector<float> host_data(N);
    cudaMemcpy(host_data.data(), d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Use std::nth_element + partial sort
    std::nth_element(host_data.begin(), host_data.begin() + (N - k), host_data.end());
    std::partial_sort(host_data.begin() + (N - k), host_data.end(), host_data.end(), std::greater<float>());

    for (int i = 0; i < k; ++i)
        h_output[i] = host_data[N - k + i];

    cudaFree(d_input);
}
