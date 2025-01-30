#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


__global__ void matMul(int* a, int* b, int* c, int n){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    int sum = 0;
    if(col < n && row < n){
        for (int i=0; i<n; i++){
            sum += a[row*n + i] + b[i*n + col];
        }
        c[row*n + col] = sum;
    }
}


int main(){
    int n = 256;
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    size_t size = n * n * sizeof(int);

    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Device memory allocation

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Initialize matrices (example values)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_a[i * n + j] = i + j;
            h_b[i * n + j] = j + 1;
        }
    }

    // Copy data to device

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_b, size, cudaMemcpyHostToDevice);

    // Kernel Config

    dim3 blockDim = {16,16,1};
    dim3 gridDim = {(unsigned int)(n + blockDim.x - 1) / blockDim.x, (unsigned int)(n + blockDim.y - 1) / blockDim.y, 1};

    // Launch Kernel

    matMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Copy result to Host

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);


    // Print results (for small matrices)

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%d ", h_c[i * n + j]);
    //     }
    //     printf("\n");
    // }

    // Free Memory

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaDeviceReset();
    return 0;

}
