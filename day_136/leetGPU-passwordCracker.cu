// password_cracker.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_LEN 6
#define CHARSET_SIZE 26
#define BASE_CHAR 'a'
#define FNV_OFFSET 2166136261U
#define FNV_PRIME 16777619U

__device__ __host__ unsigned fnv1a(const char* str, int len, int R) {
    unsigned hash = FNV_OFFSET;
    for (int r = 0; r < R; ++r) {
        for (int i = 0; i < len; ++i) {
            hash ^= (unsigned char)str[i];
            hash *= FNV_PRIME;
        }
    }
    return hash;
}

__device__ void index_to_password(unsigned long long idx, int len, char* out) {
    for (int i = len - 1; i >= 0; --i) {
        out[i] = BASE_CHAR + (idx % CHARSET_SIZE);
        idx /= CHARSET_SIZE;
    }
    out[len] = '\0';
}

__global__ void crack_kernel(unsigned target_hash, int len, int R, char* output, bool* found) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = 1;
    for (int i = 0; i < len; ++i)
        total *= CHARSET_SIZE;

    if (idx >= total || *found) return;

    char candidate[MAX_LEN + 1];
    index_to_password(idx, len, candidate);

    if (fnv1a(candidate, len, R) == target_hash) {
        if (atomicExch(found, true) == false) {
            for (int i = 0; i <= len; ++i)
                output[i] = candidate[i];
        }
    }
}

extern "C" void solve(unsigned target_hash, int password_length, int R, char* output_password) {
    char* d_output;
    bool* d_found;
    cudaMalloc(&d_output, (password_length + 1) * sizeof(char));
    cudaMalloc(&d_found, sizeof(bool));
    cudaMemset(d_found, 0, sizeof(bool));

    unsigned long long total = 1;
    for (int i = 0; i < password_length; ++i)
        total *= CHARSET_SIZE;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    crack_kernel<<<blocks, threads>>>(target_hash, password_length, R, d_output, d_found);

    cudaMemcpy(output_password, d_output, (password_length + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_found);
}
