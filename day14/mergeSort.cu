#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Number of elements
#define THREADS_PER_BLOCK 256

// **Merge function for two sorted subarrays**
__device__ void merge(int *array, int *temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    
    while (i <= mid && j <= right) {
        if (array[i] <= array[j])
            temp[k++] = array[i++];
        else
            temp[k++] = array[j++];
    }
    while (i <= mid) temp[k++] = array[i++];
    while (j <= right) temp[k++] = array[j++];

    // Copy merged elements back
    for (i = left; i <= right; i++)
        array[i] = temp[i];
}

// **CUDA Kernel for Merge Sort (bottom-up)**
__global__ void merge_sort_kernel(int *array, int *temp, int width, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int left = tid * (2 * width);
    int mid = left + width - 1;
    int right = min(left + 2 * width - 1, n - 1);

    if (mid < right)
        merge(array, temp, left, mid, right);
}

// **Host function to launch the merge sort kernel**
void merge_sort(int *h_array) {
    int *d_array, *d_temp;
    size_t bytes = N * sizeof(int);

    // Allocate memory on device
    cudaMalloc(&d_array, bytes);
    cudaMalloc(&d_temp, bytes);
    cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

    // **Iterative merge sort using CUDA**
    for (int width = 1; width < N; width *= 2) {
        int blocks = (N + 2 * width - 1) / (2 * width);
        merge_sort_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_array, d_temp, width, N);
        cudaDeviceSynchronize();
    }

    // Copy sorted array back to host
    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_array);
    cudaFree(d_temp);
}

int main() {
    int h_array[N];

    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        h_array[i] = rand() % 1000;
    }

    // Sort using CUDA Merge Sort
    merge_sort(h_array);

    // Print first 10 elements
    std::cout << "Sorted Array (First 10 elements):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
