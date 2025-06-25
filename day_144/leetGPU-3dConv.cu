// conv3d.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output,
                              int input_d, int input_r, int input_c,
                              int kernel_d, int kernel_r, int kernel_c) {
    int out_d = input_d - kernel_d + 1;
    int out_r = input_r - kernel_r + 1;
    int out_c = input_c - kernel_c + 1;

    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= out_d || r >= out_r || c >= out_c) return;

    float sum = 0.0f;
    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kr = 0; kr < kernel_r; ++kr) {
            for (int kc = 0; kc < kernel_c; ++kc) {
                int in_d = d + kd;
                int in_r = r + kr;
                int in_c = c + kc;

                int input_idx = in_d * input_r * input_c + in_r * input_c + in_c;
                int kernel_idx = kd * kernel_r * kernel_c + kr * kernel_c + kc;

                sum += input[input_idx] * kernel[kernel_idx];
            }
        }
    }

    int out_idx = d * out_r * out_c + r * out_c + c;
    output[out_idx] = sum;
}

extern "C" void solve(const float* h_input, const float* h_kernel, float* h_output,
                      int input_d, int input_r, int input_c,
                      int kernel_d, int kernel_r, int kernel_c) {
    int out_d = input_d - kernel_d + 1;
    int out_r = input_r - kernel_r + 1;
    int out_c = input_c - kernel_c + 1;

    size_t input_sz = input_d * input_r * input_c * sizeof(float);
    size_t kernel_sz = kernel_d * kernel_r * kernel_c * sizeof(float);
    size_t output_sz = out_d * out_r * out_c * sizeof(float);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_sz);
    cudaMalloc(&d_kernel, kernel_sz);
    cudaMalloc(&d_output, output_sz);

    cudaMemcpy(d_input, h_input, input_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_sz, cudaMemcpyHostToDevice);

    dim3 threads(8, 8, 8);
    dim3 blocks((out_c + 7) / 8, (out_r + 7) / 8, (out_d + 7) / 8);
    conv3d_kernel<<<blocks, threads>>>(d_input, d_kernel, d_output,
                                       input_d, input_r, input_c,
                                       kernel_d, kernel_r, kernel_c);

    cudaMemcpy(h_output, d_output, output_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
