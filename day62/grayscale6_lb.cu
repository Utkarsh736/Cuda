// rgb_to_grayscale.cu

#include <cuda_runtime.h>
#include <stdio.h> // For error messages

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
do {                                                                \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)

/**
 * @brief CUDA kernel to convert an RGB image to grayscale.
 *
 * Each thread processes one pixel (H, W).
 * Assumes input RGB values are floats in [0, 1].
 * Uses the formula: Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
 *
 * @param rgb_input Pointer to the input RGB image data on the GPU (H, W, 3).
 * @param gray_output Pointer to the output grayscale image data on the GPU (H, W).
 * @param H Height of the image.
 * @param W Width of the image.
 */
__global__ void rgb_to_grayscale_kernel(const float* __restrict__ rgb_input,
                                        float* __restrict__ gray_output,
                                        int H, int W)
{
    // Calculate the global pixel coordinates (column and row) for this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the calculated coordinates are within the image bounds
    if (row < H && col < W) {
        // Calculate the base index for the RGB pixel in the flattened input array
        // Input layout is assumed to be HWC (Height, Width, Channel)
        int rgb_idx = (row * W + col) * 3;

        // Calculate the index for the grayscale pixel in the flattened output array
        int gray_idx = row * W + col;

        // Read the R, G, B values from global memory
        float r = rgb_input[rgb_idx + 0];
        float g = rgb_input[rgb_idx + 1];
        float b = rgb_input[rgb_idx + 2];

        // Apply the grayscale conversion formula
        // Use 'f' suffix for float literals for clarity and potential minor optimization
        float gray_val = 0.2989f * r + 0.5870f * g + 0.1140f * b;

        // Write the calculated grayscale value to global memory
        gray_output[gray_idx] = gray_val;
    }
}

/**
 * @brief Host function to manage memory and launch the RGB to grayscale kernel.
 *
 * @param h_rgb_input Pointer to the input RGB image data on the host (CPU).
 * @param h_gray_output Pointer to the output grayscale image data buffer on the host (CPU).
 * @param H Height of the image (must be even and H=W).
 * @param W Width of the image (must be even and H=W).
 */
void convert_rgb_to_grayscale_cuda(const float* h_rgb_input, float* h_gray_output, int H, int W) {
    // --- Input Validation (Optional but Recommended) ---
    if (H <= 0 || W <= 0 || H != W || H % 2 != 0) {
        fprintf(stderr, "Error: Image dimensions must be square, positive, and even (H=%d, W=%d).\n", H, W);
        exit(EXIT_FAILURE);
    }
    if (h_rgb_input == nullptr || h_gray_output == nullptr) {
         fprintf(stderr, "Error: Host input or output pointer is null.\n");
        exit(EXIT_FAILURE);
    }

    // --- Calculate Memory Sizes ---
    size_t rgb_size_bytes = (size_t)H * W * 3 * sizeof(float);
    size_t gray_size_bytes = (size_t)H * W * sizeof(float);

    // --- Allocate GPU Memory ---
    float *d_rgb_input = nullptr;
    float *d_gray_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_rgb_input, rgb_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gray_output, gray_size_bytes));

    // --- Copy Input Data from Host to Device ---
    CUDA_CHECK(cudaMemcpy(d_rgb_input, h_rgb_input, rgb_size_bytes, cudaMemcpyHostToDevice));

    // --- Configure Kernel Launch Parameters ---
    // Use 2D blocks, common sizes are 16x16 or 32x32. 16x16=256 threads/block is often a safe choice.
    dim3 blockSize(16, 16);
    // Calculate grid dimensions needed to cover the entire image
    // Ceiling division: (N + M - 1) / M
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x,
                  (H + blockSize.y - 1) / blockSize.y);

    // --- Launch the Kernel ---
    rgb_to_grayscale_kernel<<<gridSize, blockSize>>>(d_rgb_input, d_gray_output, H, W);

    // --- Check for Kernel Launch Errors ---
    // cudaPeekAtLastError() checks for asynchronous errors from the kernel launch
    CUDA_CHECK(cudaPeekAtLastError());
    // cudaDeviceSynchronize() waits for the kernel to complete and returns any execution errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copy Output Data from Device to Host ---
    CUDA_CHECK(cudaMemcpy(h_gray_output, d_gray_output, gray_size_bytes, cudaMemcpyDeviceToHost));

    // --- Free GPU Memory ---
    CUDA_CHECK(cudaFree(d_rgb_input));
    CUDA_CHECK(cudaFree(d_gray_output));
}

// --- Example Usage (Optional - requires linking with a main function) ---
/*
#include <vector>
#include <iostream>

int main() {
    int H = 256; // Example size (square and even)
    int W = 256;

    // Allocate host memory
    std::vector<float> h_rgb(H * W * 3);
    std::vector<float> h_gray(H * W);

    // Initialize input with dummy data (e.g., a gradient)
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int idx = (r * W + c) * 3;
            h_rgb[idx + 0] = (float)c / (W - 1); // R = horizontal gradient
            h_rgb[idx + 1] = (float)r / (H - 1); // G = vertical gradient
            h_rgb[idx + 2] = 0.5f;               // B = constant
        }
    }

    // Call the CUDA function
    convert_rgb_to_grayscale_cuda(h_rgb.data(), h_gray.data(), H, W);

    std::cout << "CUDA RGB to Grayscale conversion completed successfully." << std::endl;

    // Optional: Print a few output values to verify
    // std::cout << "Sample output grayscale values:" << std::endl;
    // for(int i=0; i < 5; ++i) {
    //     std::cout << h_gray[i] << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
*/
