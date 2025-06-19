// gaussian_blur.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gaussian_blur_kernel(const float* image, const float* kernel,
                                     float* output, int rows, int cols,
                                     int k_rows, int k_cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= rows || c >= cols) return;

    int k_center_r = k_rows / 2;
    int k_center_c = k_cols / 2;

    float sum = 0.0f;
    for (int i = 0; i < k_rows; ++i) {
        for (int j = 0; j < k_cols; ++j) {
            int img_r = r + i - k_center_r;
            int img_c = c + j - k_center_c;

            if (img_r >= 0 && img_r < rows && img_c >= 0 && img_c < cols) {
                float img_val = image[img_r * cols + img_c];
                float k_val = kernel[i * k_cols + j];
                sum += img_val * k_val;
            }
        }
    }
    output[r * cols + c] = sum;
}

extern "C" void solve(const float* h_image, const float* h_kernel, float* h_output,
                      int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols) {
    float *d_image, *d_kernel, *d_output;
    size_t image_size = input_rows * input_cols * sizeof(float);
    size_t kernel_size = kernel_rows * kernel_cols * sizeof(float);

    cudaMalloc(&d_image, image_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, image_size);

    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((input_cols + 15) / 16, (input_rows + 15) / 16);

    gaussian_blur_kernel<<<blocks, threads>>>(d_image, d_kernel, d_output,
                                              input_rows, input_cols,
                                              kernel_rows, kernel_cols);
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
