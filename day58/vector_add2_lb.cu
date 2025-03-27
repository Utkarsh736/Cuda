#include <torch/extension.h>
#include <cuda_fp16.h>

// CUDA kernel for float16 vector addition
__global__ void vector_add_kernel(const half* a, const half* b, half* c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        c[idx] = __hadd(a[idx], b[idx]);  // Half-precision addition
    }
}

// Host function to launch the kernel
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.dim() == 2, "Input tensor a must be 2D");
    TORCH_CHECK(b.dim() == 2, "Input tensor b must be 2D");
    TORCH_CHECK(a.size(0) == b.size(0) && a.size(1) == b.size(1), "Input tensors must have the same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat16, "Input tensor a must be of type torch.float16");
    TORCH_CHECK(b.dtype() == torch::kFloat16, "Input tensor b must be of type torch.float16");

    // Extract dimensions
    const auto height = a.size(0);
    const auto width = a.size(1);

    // Allocate output tensor
    auto c = torch::empty({height, width}, a.options().dtype(torch::kFloat16));

    // Define block and grid dimensions
    const int BLOCK_SIZE = 16;
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((width + threads.x - 1) / threads.x,
                      (height + threads.y - 1) / threads.y);

    // Launch the kernel
    vector_add_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(a.data_ptr()),
        reinterpret_cast<half*>(b.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        width,
        height
    );

    return c;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vector_add_cuda, "Vector addition (CUDA)");
}
