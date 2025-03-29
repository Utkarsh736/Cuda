import numpy as np
from numba import cuda, float32
import math
import cv2

# Define the CUDA kernel
@cuda.jit
def rgb_to_grayscale_kernel(rgb_image, grayscale_image):
    # Define the standard coefficients for RGB to Grayscale conversion
    coef_r = 0.2989
    coef_g = 0.5870
    coef_b = 0.1140

    # Calculate the position of the current thread
    x, y = cuda.grid(2)

    # Get the dimensions of the image
    height, width, _ = rgb_image.shape

    if x < width and y < height:
        # Extract the RGB components
        r = rgb_image[y, x, 0]
        g = rgb_image[y, x, 1]
        b = rgb_image[y, x, 2]

        # Compute the grayscale value
        gray = coef_r * r + coef_g * g + coef_b * b

        # Store the result in the grayscale image
        grayscale_image[y, x] = gray

# Function to convert an RGB image to grayscale using the defined CUDA kernel
def convert_rgb_to_grayscale(rgb_image):
    # Ensure the input image is in float32 format for processing
    rgb_image = rgb_image.astype(np.float32) / 255.0

    # Get the dimensions of the image
    height, width, channels = rgb_image.shape

    # Allocate memory for the output grayscale image
    grayscale_image = np.empty((height, width), dtype=np.float32)

    # Define the number of threads per block and the number of blocks per grid
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    rgb_to_grayscale_kernel[blocks_per_grid, threads_per_block](rgb_image, grayscale_image)

    # Wait for the GPU to finish before accessing the host
    cuda.synchronize()

    return grayscale_image

# Example usage
if __name__ == "__main__":
    # Load an RGB image using OpenCV
    input_image = cv2.imread('input_image.jpg')
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale using the CUDA-accelerated function
    grayscale_image = convert_rgb_to_grayscale(rgb_image)

    # Convert the grayscale image to uint8 format and save it
    grayscale_image_uint8 = (grayscale_image * 255).astype(np.uint8)
    cv2.imwrite('grayscale_image.jpg', grayscale_image_uint8)
