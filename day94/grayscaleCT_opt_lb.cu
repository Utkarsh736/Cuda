//!POPCORN leaderboard grayscale

#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int TILE = 32;  // Block dimension (32×32 threads)

// Kernel: vectorized loads + coalesced writes + minimal divergence
__global__ void grayscale_kernel(const float4* __restrict__ rgb4,
                                 float*       __restrict__ gray,
                                 int width, int height) {
    // Compute pixel (x,y) for this thread
    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x,      ty = threadIdx.y;
    int x0 = bx + tx*2, y = by + ty;  // two pixels per thread horizontally

    if (y >= height) return;

    // Load two RGB pixels as float4 (R,G,B,A) where A unused
    if (x0 + 1 < width) {
        float4 p = rgb4[y * (width/2) + tx];  // one float4 covers two pixels 0

        // Unpack first pixel
        float r1 = p.x, g1 = p.y, b1 = p.z;
        // Compute grayscale
        gray[y*width + x0] = 0.2989f*r1 + 0.5870f*g1 + 0.1140f*b1;  1

        // Unpack second pixel
        float r2 = p.w, g2 = __ldg(((const float*)&p)[4]), b2 = __ldg(((const float*)&p)[5]);
        gray[y*width + x0+1] = 0.2989f*r2 + 0.5870f*g2 + 0.1140f*b2; 2
    } else if (x0 < width) {
        // Handle last odd pixel
        float4 p = rgb4[y * ((width+1)/2) + tx];
        float r = p.x, g = p.y, b = p.z;
        gray[y*width + x0] = 0.2989f*r + 0.5870f*g + 0.1140f*b;
    }
}

// Host binding for PyTorch
torch::Tensor custom_kernel(torch::Tensor input) {
    auto opts = torch::TensorOptions()
                    .dtype(input.dtype())
                    .device(input.device());
    int height = input.size(0), width = input.size(1);
    // Allocate output
    auto output = torch::empty({height, width}, opts);

    // Reinterpret input as float4 pointer for vectorized loads
    const float4* d_in4 = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float*         d_out = output.data_ptr<float>();

    // Launch configuration
    dim3 block(TILE/2, TILE);  // each thread loads 2 pixels horizontally 3
    dim3 grid((width + TILE -1)/TILE, (height+TILE-1)/TILE);

    grayscale_kernel<<<grid, block>>>(d_in4, d_out, width, height);
    cudaDeviceSynchronize();  // ensure completion 4

    return output;
}

// PyBind for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_kernel", &custom_kernel, "Optimized Grayscale (CUDA)");
}