#include <iostream>
#include <cuda_runtime.h>

#define N 16  // Must be a power of 2
#define THREADS_PER_BLOCK 16

__device__ void swap(int &a, int &b, bool ascending) {
    if ((a > b) == ascending) {
        int temp = a;
        a = b;
        b = temp;
    }
}

// **Bitonic Sorting Kernel**
__global__ void bitonic_sort(int *arr, int j, int k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = tid ^ j;  // XOR for bitonic comparison

    if (ixj > tid) {
        bool ascending = ((tid & k) == 0);
        swap(arr[tid], arr[ixj], ascending);
    }
}

void host_bitonic_sort(int *arr) {
    int *d_arr;
    size_t size = N * sizeof(int);

    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // Bitonic sorting network execution
    for (int k = 2; k <= N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonic_sort<<<1, N>>>(d_arr, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int arr[N] = {14, 2, 15, 8, 9, 4, 6, 1, 11, 3, 5, 13, 7, 12, 10, 0};

    std::cout << "Before Sorting:\n";
    for (int i = 0; i < N; i++) std::cout << arr[i] << " ";
    std::cout << "\n";

    host_bitonic_sort(arr);

    std::cout << "After Sorting:\n";
    for (int i = 0; i < N; i++) std::cout << arr[i] << " ";
    std::cout << "\n";

    return 0;
}
