import torch
import torch.utils.cpp_extension
import time
import os

# --- CUDA Kernel Definition ---
# Note: CUDA uses 'half' for float16
cuda_source = """
#include <cuda_fp16.h> // Required for half type

extern "C" __global__
void fp16_add_kernel(const half* __restrict__ a,
                     const half* __restrict__ b,
                     half* __restrict__ c,
                     int N) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure threads do not access out of bounds memory
    if (idx < N && idy < N) {
        // Calculate the linear index for the 2D array
        int index = idy * N + idx;
        // Perform the float16 addition
        c[index] = __hadd(a[index], b[index]); // Use __hadd for half precision addition
    }
}
"""

# --- C++ Wrapper Definition ---
# This C++ code acts as an interface between Python/PyTorch and the CUDA kernel.
cpp_source = """
#include <torch/extension.h>
#include <cuda_fp16.h> // Also needed here for ATen/PyTorch types

// Forward declaration of the CUDA kernel
void fp16_add_kernel(const half* a, const half* b, half* c, int N);

// C++ interface function callable from Python
torch::Tensor fp16_add_torch(torch::Tensor a, torch::Tensor b) {
    // Input validation (basic checks)
    TORCH_CHECK(a.device().is_cuda(), "Input tensor 'a' must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor 'b' must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "Input tensor 'a' must be float16");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "Input tensor 'b' must be float16");
    TORCH_CHECK(a.dim() == 2, "Input tensor 'a' must be 2D");
    TORCH_CHECK(b.dim() == 2, "Input tensor 'b' must be 2D");
    TORCH_CHECK(a.size(0) == a.size(1), "Input tensor 'a' must be square (N, N)");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(a.is_contiguous(), "Input tensor 'a' must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor 'b' must be contiguous");

    int N = a.size(0); // Assuming square N x N matrix

    // Create the output tensor (same shape, type, and device as input 'a')
    torch::Tensor c = torch::empty_like(a);

    // Define CUDA grid and block dimensions
    // Usually 16x16 or 32x32 are good starting points for block size
    const dim3 threads_per_block(16, 16);
    const dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                          (N + threads_per_block.y - 1) / threads_per_block.y);

    // Launch the CUDA kernel
    fp16_add_kernel<<<num_blocks, threads_per_block>>>(
        (const half*)a.data_ptr<at::Half>(), // Get raw pointer cast to half*
        (const half*)b.data_ptr<at::Half>(), // Get raw pointer cast to half*
        (half*)c.data_ptr<at::Half>(),       // Get raw pointer cast to half*
        N
    );

    // Check for CUDA errors after kernel launch (optional but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return c;
}
"""

# --- Load the CUDA/C++ Extension ---
# This compiles the CUDA and C++ code and loads it as a Python module.
# It might take a moment the first time it's run.
# Set TORCH_EXTENSIONS_DIR environment variable if you want to control build location
print("Compiling and loading CUDA extension...")
start_compile = time.time()
try:
    fp16_add_extension = torch.utils.cpp_extension.load_inline(
        name='fp16_add_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['fp16_add_torch'], # Function to expose to Python
        verbose=True # Set to False to reduce compilation output
    )
    print(f"Compilation successful! Took {time.time() - start_compile:.2f} seconds.")
except Exception as e:
    print(f"ERROR: Failed to compile or load the CUDA extension.")
    print(e)
    # Provide hints if common issues occur
    if "nvcc" in str(e).lower() or "cuda" in str(e).lower():
        print("Hint: Ensure the NVIDIA CUDA Toolkit (nvcc) is installed and in your system's PATH.")
        print("      Verify PyTorch was installed with CUDA support.")
    exit() # Exit if compilation fails


# --- Python Function using the Extension ---
def fp16_vector_add_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise addition of two float16 tensors on the GPU
    using the custom CUDA kernel.

    Args:
        a: First input tensor (N, N), float16, on CUDA device.
        b: Second input tensor (N, N), float16, on CUDA device.

    Returns:
        Output tensor (N, N), float16, on CUDA device containing a + b.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")
    if not a.is_cuda or not b.is_cuda:
        raise TypeError("Input tensors must be on the GPU (CUDA device). Use .to('cuda')")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
         raise TypeError("Input tensors must be of type torch.float16.")
    if a.shape != b.shape or len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input tensors must be square (N, N) and have the same shape.")

    # Ensure tensors are contiguous in memory - important for raw pointer access
    a_cont = a.contiguous()
    b_cont = b.contiguous()

    # Call the C++ function from the loaded extension
    return fp16_add_extension.fp16_add_torch(a_cont, b_cont)

# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA device not found. Skipping GPU execution.")
    else:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")

        # --- Parameters ---
        N = 1024  # Size of the square matrix (e.g., 1024x1024)

        # --- Input Data Generation ---
        print(f"\nGenerating input tensors of shape ({N}, {N}) with dtype torch.float16...")
        # Create tensors on CPU first for clarity, then move to GPU
        a_cpu = torch.randn(N, N, dtype=torch.float32) # Generate as float32 first
        b_cpu = torch.randn(N, N, dtype=torch.float32)

        # Convert to float16 and move to GPU
        a_gpu = a_cpu.to(dtype=torch.float16, device=device)
        b_gpu = b_cpu.to(dtype=torch.float16, device=device)
        print("Input tensors moved to GPU.")

        # --- Execution ---

        # 1. Using the Custom CUDA Kernel
        print("\nRunning addition using custom CUDA kernel...")
        # Warm-up GPU (optional, but good for accurate timing)
        _ = fp16_vector_add_gpu(a_gpu, b_gpu)
        torch.cuda.synchronize() # Wait for GPU operations to complete
        start_time_custom = time.time()
        c_custom_gpu = fp16_vector_add_gpu(a_gpu, b_gpu)
        torch.cuda.synchronize()
        end_time_custom = time.time()
        print(f"Custom CUDA kernel execution time: {(end_time_custom - start_time_custom)*1000:.4f} ms")

        # 2. Using PyTorch's built-in addition (for comparison)
        print("\nRunning addition using PyTorch built-in '+' operator...")
         # Warm-up GPU
        _ = a_gpu + b_gpu
        torch.cuda.synchronize()
        start_time_pytorch = time.time()
        c_pytorch_gpu = a_gpu + b_gpu # PyTorch's built-in, GPU-accelerated add
        torch.cuda.synchronize()
        end_time_pytorch = time.time()
        print(f"PyTorch built-in '+' execution time: {(end_time_pytorch - start_time_pytorch)*1000:.4f} ms")

        # --- Verification ---
        print("\nVerifying results...")
        # Move results back to CPU for comparison if needed, or compare on GPU
        # Comparing on GPU is generally preferred to avoid data transfer overhead

        # Use torch.allclose for floating-point comparisons
        # Float16 has lower precision, so adjust tolerances if necessary
        # Default tolerances for float16 might be sufficient often.
        # atol (absolute tolerance), rtol (relative tolerance)
        are_close = torch.allclose(c_custom_gpu, c_pytorch_gpu, atol=1e-3, rtol=1e-3)

        if are_close:
            print("SUCCESS: Results from custom kernel and PyTorch built-in '+' match!")
        else:
            print("FAILURE: Results from custom kernel and PyTorch built-in '+' DO NOT match.")
            # Optionally print differences for debugging:
            diff = torch.abs(c_custom_gpu - c_pytorch_gpu)
            print(f"Maximum difference: {torch.max(diff)}")
            print(f"Mean difference: {torch.mean(diff.float())}") # Cast to float32 for mean

        # --- Optional: Inspect Output ---
        # print("\nSample of custom kernel output (first 5x5 elements):")
        # print(c_custom_gpu[:5, :5])
        # print("\nSample of PyTorch output (first 5x5 elements):")
        # print(c_pytorch_gpu[:5, :5])
