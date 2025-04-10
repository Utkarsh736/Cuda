import torch
import triton
import triton.language as tl
import sys # Required for eval tests if they check stdout/stderr

# --- Triton Kernel ---

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to the first input tensor
    y_ptr,  # Pointer to the second input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Total number of elements in the tensors (N*N)
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program instance processes
):
    """
    Triton kernel for element-wise addition of two tensors (float16).

    Processes `n_elements` in blocks of size `BLOCK_SIZE`.
    """
    # 1. Calculate the offsets for the current program instance
    # Each program instance handles a unique block of data.
    pid = tl.program_id(axis=0)  # Get the unique ID for this instance (0..num_programs-1)
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets for the elements this instance will process
    # e.g., if pid=0, offsets = [0, 1, ..., BLOCK_SIZE-1]
    #       if pid=1, offsets = [BLOCK_SIZE, ..., 2*BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 2. Create a mask to handle potential out-of-bounds memory access
    # This is crucial if `n_elements` is not perfectly divisible by `BLOCK_SIZE`.
    # The mask ensures we only load/store valid elements within the tensor bounds.
    mask = offsets < n_elements

    # 3. Load the input data from global memory (HBM) to SRAM
    # `mask=mask` ensures that loads outside the valid range are handled safely
    # (Triton typically handles this by loading a default value, often 0,
    # for masked-out elements, which is fine for addition).
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 4. Perform the element-wise addition in SRAM
    # This computation happens on the loaded blocks.
    output = x + y

    # 5. Store the result from SRAM back to global memory (HBM)
    # `mask=mask` ensures that we only write results for valid elements.
    tl.store(output_ptr + offsets, output, mask=mask)


# --- Python Wrapper Function ---

def vector_add(input_tuple: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Implements float16 vector addition using a Triton kernel.

    Input: tuple(torch.Tensor, torch.Tensor) with tensors of shape (N, N)
           and type torch.float16. These tensors are assumed to be on a CUDA device.
    Output: torch.Tensor of shape (N, N) and type torch.float16 containing the sum.
    """
    x, y = input_tuple

    # --- Input Validation (Good Practice) ---
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("Inputs must be torch.Tensors")
    if x.shape != y.shape:
        raise ValueError(f"Input tensors must have the same shape, got {x.shape} and {y.shape}")
    # Although the spec says (N, N), numel() allows flexibility for flattened or other shapes
    # if len(x.shape) != 2:
    #     raise ValueError(f"Input tensors must be 2D (N, N), got shape {x.shape}")
    if x.dtype != torch.float16 or y.dtype != torch.float16:
        raise TypeError(f"Input tensors must be torch.float16, got {x.dtype} and {y.dtype}")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")

    # --- Output Tensor Allocation ---
    # Create an output tensor with the same shape, dtype, and device as the inputs.
    output = torch.empty_like(x)

    # --- Kernel Launch Parameters ---
    n_elements = output.numel() # Total number of elements (N*N)

    # `BLOCK_SIZE` is a tunable parameter. Needs to be a power of 2.
    # Affects performance. 1024 is often a reasonable starting point.
    # Larger blocks can increase occupancy but might require more SRAM.
    BLOCK_SIZE = 1024

    # `grid` determines the number of program instances (kernel launches).
    # We need enough instances to cover all `n_elements`.
    # `triton.cdiv(a, b)` computes ceiling division (a + b - 1) // b.
    # We launch a 1D grid of instances.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # --- Kernel Execution ---
    # Call the Triton kernel.
    # The `[grid]` syntax specifies the launch grid.
    # Arguments are passed to the kernel function.
    # Pointers to tensors are implicitly handled by Triton when passing torch tensors.
    # `BLOCK_SIZE` is passed as a compile-time constant (`constexpr`).
    add_kernel[grid](
        x,  # Pass tensor x (becomes x_ptr)
        y,  # Pass tensor y (becomes y_ptr)
        output,  # Pass output tensor (becomes output_ptr)
        n_elements,  # Pass the total number of elements
        BLOCK_SIZE=BLOCK_SIZE, # Pass block size as compile-time constant
    )

    # --- Return Result ---
    return output

# --- Example Usage / Test Stub for Eval ---
# This part demonstrates how the function might be called and verified.
# Evaluation systems might import the `vector_add` function and run similar tests.
if __name__ == "__main__":
    print("Running Triton vector_add example...")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA device not found. Skipping execution.", file=sys.stderr)
        sys.exit(0) # Exit cleanly if no CUDA

    # --- Configuration ---
    N = 1024  # Example dimension for the square tensor
    device = 'cuda'
    dtype = torch.float16

    # --- Create Input Data ---
    # Create tensors matching the description (normal distribution, float16, on CUDA)
    print(f"Creating input tensors of shape ({N}, {N}), dtype={dtype}, device='{device}'")
    try:
        x = torch.randn((N, N), device=device, dtype=dtype)
        y = torch.randn((N, N), device=device, dtype=dtype)
        input_data = (x, y)
    except Exception as e:
        print(f"Error creating input tensors: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Execute Triton Kernel ---
    print("Executing Triton vector_add kernel...")
    try:
        output_triton = vector_add(input_data)
    except Exception as e:
        print(f"Error executing Triton kernel: {e}", file=sys.stderr)
        # Potentially print more debug info if needed
        # print(f"Input shapes: {x.shape}, {y.shape}")
        # print(f"Input dtypes: {x.dtype}, {y.dtype}")
        # print(f"Device: {x.device}")
        sys.exit(1)

    # --- Execute Reference PyTorch Implementation ---
    print("Executing PyTorch reference addition...")
    output_torch = x + y

    # --- Verification ---
    print("Verifying results...")
    # Use torch.allclose for robust floating-point comparisons.
    # Tolerances (atol, rtol) might need adjustment based on the required precision.
    # float16 has lower precision, so tolerances shouldn't be excessively tight.
    try:
        are_close = torch.allclose(output_triton, output_torch, atol=1e-3, rtol=1e-2)
        print(f"Triton output matches PyTorch output: {are_close}")

        if not are_close:
            # Optional: Calculate and print the difference for debugging
            diff = torch.abs(output_triton - output_torch)
            print(f"Max absolute difference: {torch.max(diff).item()}")
            print(f"Mean absolute difference: {torch.mean(diff).item()}")
            # Optionally print some elements for visual inspection if needed
            # print("Sample Triton Output:\n", output_triton[:4, :4])
            # print("Sample PyTorch Output:\n", output_torch[:4, :4])
            # print("Sample Difference:\n", diff[:4, :4])
            print("Verification failed.", file=sys.stderr)
            # sys.exit(1) # Optional: exit with error if verification fails

    except Exception as e:
        print(f"Error during verification: {e}", file=sys.stderr)
        sys.exit(1)

    print("Example execution finished successfully.")
