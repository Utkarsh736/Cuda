#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLUR_SIZE 1  // (3x3 box filter)


__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sum = 0;
        int count = 0;

        // Apply 3x3 box blur
        for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; dy++) {
            for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }

        output[y * width + x] = sum / count;  
    }
}


void blurImage(unsigned char *h_input, unsigned char *h_output, int width, int height) {
    unsigned char *d_input, *d_output;

    size_t size = width * height * sizeof(unsigned char);

    
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Main function (dummy example)
int main() {
    int width = 512, height = 512;
    unsigned char *h_input = (unsigned char*)malloc(width * height);
    unsigned char *h_output = (unsigned char*)malloc(width * height);

    // Initialize input image with dummy data
    for (int i = 0; i < width * height; i++)
        h_input[i] = rand() % 256; // Random grayscale pixel values

    // Apply blur
    blurImage(h_input, h_output, width, height);

    
    free(h_input);
    free(h_output);

    printf("Image blurring completed!\n");
    return 0;
}
