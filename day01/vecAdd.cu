#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<n)C[i] = A[i] + B[i];
}

void vecAdd(float* A, float* B, float* C, int n){
    int size = n*sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (int)ceil((double)n / blockSize);
    vecAddKernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory of A,B,C
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main(){
    const int n = 1024; // Example size
    float *h_A = (float *)malloc(n * sizeof(float));
    float *h_B = (float *)malloc(n * sizeof(float));
    float *h_C = (float *)malloc(n * sizeof(float));

    // Initialize h_A and h_B with some values
    for(int i = 0; i < n; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // Call the vector addition function
    vecAdd(h_A, h_B, h_C, n);

    // Print the first few results to verify correctness
    for(int i = 0; i < 10; ++i) {
        printf("h_C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
