#include <cuda_runtime.h>
#include <iostream>

__global__ void invertColorsKernel(unsigned char* image, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        int i = idx * 4;
        image[i]     = 255 - image[i];     // R
        image[i + 1] = 255 - image[i + 1]; // G
        image[i + 2] = 255 - image[i + 2]; // B
        // image[i + 3] remains unchanged (Alpha)
    }
}

void solve(unsigned char* image, int width, int height) {
    int totalPixels = width * height;
    int totalBytes = totalPixels * 4 * sizeof(unsigned char);

    unsigned char* d_image;
    cudaMalloc((void**)&d_image, totalBytes);
    cudaMemcpy(d_image, image, totalBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    invertColorsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, totalPixels);

    cudaMemcpy(image, d_image, totalBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
}
