from math import floor

fn solve(
    image: ptr[float32], input_rows: Int, input_cols: Int,
    kernel: ptr[float32], kernel_rows: Int, kernel_cols: Int,
    output: ptr[float32]
):
    # Calculate padding
    let pad_rows = kernel_rows // 2
    let pad_cols = kernel_cols // 2

    for i in range(input_rows):
        for j in range(input_cols):
            var sum: float32 = 0.0

            for ki in range(kernel_rows):
                for kj in range(kernel_cols):
                    # Compute input image coordinates with offset
                    let ii = i + ki - pad_rows
                    let jj = j + kj - pad_cols

                    # Boundary check (zero padding)
                    if 0 <= ii < input_rows and 0 <= jj < input_cols:
                        let img_val = image[ii * input_cols + jj]
                        let ker_val = kernel[ki * kernel_cols + kj]
                        sum += img_val * ker_val

            output[i * input_cols + j] = sum
