#include <cuda_runtime.h>
#include <cmath>

// Kernel to find the maximum value in the input array
__global__ void find_max(const float* input, float* max_val, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float val = -INFINITY;

    // Load elements into shared memory
    if (idx < N) val = input[idx];
    sdata[tid] = val;
    __syncthreads();

    // Perform parallel reduction to find the maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) max_val[blockIdx.x] = sdata[0];
}

// Kernel to compute exponentials and their sum
__global__ void compute_exp_sum(const float* input, float* exp_output, float* sum_exp, float max_val, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float val = 0.0f;

    // Compute exponentials
    if (idx < N) {
        val = expf(input[idx] - max_val);
        exp_output[idx] = val;
    }
    sdata[tid] = val;
    __syncthreads();

    // Perform parallel reduction to compute the sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) sum_exp[blockIdx.x] = sdata[0];
}

// Kernel to compute the final softmax output
__global__ void compute_softmax(float* exp_output, float sum_exp, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = exp_output[idx] / sum_exp;
    }
}

void solve(const float* input, float* output, int N) {
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float *d_input, *d_exp_output, *d_max_vals, *d_sum_vals;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_exp_output, N * sizeof(float));
    cudaMalloc(&d_max_vals, blocks * sizeof(float));
    cudaMalloc(&d_sum_vals, blocks * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: Find the maximum value
    find_max<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_max_vals, N);

    // Copy max values back to host and find the overall maximum
    float* h_max_vals = new float[blocks];
    cudaMemcpy(h_max_vals, d_max_vals, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float max_val = h_max_vals[0];
    for (int i = 1; i < blocks; ++i) {
        if (h_max_vals[i] > max_val) max_val = h_max_vals[i];
    }
    delete[] h_max_vals;

    // Step 2: Compute exponentials and their sum
    compute_exp_sum<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_exp_output, d_sum_vals, max_val, N);

    // Copy sum values back to host and compute the total sum
    float* h_sum_vals = new float[blocks];
    cudaMemcpy(h_sum_vals, d_sum_vals, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float sum_exp = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        sum_exp += h_sum_vals[i];
    }
    delete[] h_sum_vals;

    // Step 3: Compute the final softmax output
    compute_softmax<<<blocks, threads>>>(d_exp_output, sum_exp, output, N);

    // Free allocated memory
    cudaFree(d_input);
    cudaFree(d_exp_output);
    cudaFree(d_max_vals);
    cudaFree(d_sum_vals);
}
