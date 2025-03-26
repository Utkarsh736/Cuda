#include <torch/extension.h>

// CUDA kernel to convert RGB to grayscale
__global__ void grayscale_kernel(const float* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // Index in RGB array (3 channels per pixel)
        gray[y * width + x] = 0.2989f * rgb[idx]     // R
                            + 0.5870f * rgb[idx + 1] // G
                            + 0.1140f * rgb[idx + 2]; // B
    }
}

// Host function to manage kernel launch
torch::Tensor grayscale_cuda_forward(torch::Tensor rgb) {
    // Input validation
    TORCH_CHECK(rgb.dim() == 3, "Input must be a 3D tensor");
    TORCH_CHECK(rgb.size(2) == 3, "Input must have 3 channels");

    // Ensure tensor is contiguous in memory
    rgb = rgb.contiguous();

    // Extract dimensions
    const auto height = rgb.size(0);
    const auto width = rgb.size(1);

    // Allocate output tensor
    auto gray = torch::empty({height, width}, rgb.options().dtype(torch::kFloat32));

    // Define block and grid dimensions
    const int BLOCK_SIZE = 16;
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    // Launch the kernel
    grayscale_kernel<<<blocks, threads>>>(
        rgb.data_ptr<float>(),
        gray.data_ptr<float>(),
        width,
        height
    );

    return gray;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grayscale_cuda_forward, "Grayscale forward (CUDA)");
}
