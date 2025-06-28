// kmeans_2d.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define THREADS 256
#define THRESHOLD 0.0001f

__device__ float distance2(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy;
}

__global__ void assign_labels(const float* data_x, const float* data_y,
                              const float* centroids_x, const float* centroids_y,
                              int* labels, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float min_dist = 1e20f;
    int best_k = 0;
    for (int j = 0; j < k; ++j) {
        float dist = distance2(data_x[i], data_y[i], centroids_x[j], centroids_y[j]);
        if (dist < min_dist) {
            min_dist = dist;
            best_k = j;
        }
    }
    labels[i] = best_k;
}

__global__ void clear_centroids(float* sum_x, float* sum_y, int* count, int k) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < k) {
        sum_x[i] = 0.0f;
        sum_y[i] = 0.0f;
        count[i] = 0;
    }
}

__global__ void accumulate_centroids(const float* data_x, const float* data_y,
                                     const int* labels, float* sum_x,
                                     float* sum_y, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cluster = labels[i];
    atomicAdd(&sum_x[cluster], data_x[i]);
    atomicAdd(&sum_y[cluster], data_y[i]);
    atomicAdd(&count[cluster], 1);
}

__global__ void update_centroids(float* centroids_x, float* centroids_y,
                                 const float* sum_x, const float* sum_y,
                                 const int* count, float* diff, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k || count[i] == 0) return;

    float new_x = sum_x[i] / count[i];
    float new_y = sum_y[i] / count[i];

    float dx = new_x - centroids_x[i];
    float dy = new_y - centroids_y[i];
    diff[i] = dx * dx + dy * dy;

    centroids_x[i] = new_x;
    centroids_y[i] = new_y;
}

extern "C" void solve(const float* h_data_x, const float* h_data_y,
                      float* h_centroids_x, float* h_centroids_y,
                      int* h_labels, int sample_size, int k, int max_iterations) {
    float *d_data_x, *d_data_y, *d_centroids_x, *d_centroids_y;
    float *d_sum_x, *d_sum_y, *d_diff;
    int *d_labels, *d_count;

    size_t nf = sample_size * sizeof(float);
    size_t kf = k * sizeof(float);
    size_t ni = sample_size * sizeof(int);
    size_t ki = k * sizeof(int);

    cudaMalloc(&d_data_x, nf);
    cudaMalloc(&d_data_y, nf);
    cudaMalloc(&d_centroids_x, kf);
    cudaMalloc(&d_centroids_y, kf);
    cudaMalloc(&d_labels, ni);
    cudaMalloc(&d_sum_x, kf);
    cudaMalloc(&d_sum_y, kf);
    cudaMalloc(&d_count, ki);
    cudaMalloc(&d_diff, kf);

    cudaMemcpy(d_data_x, h_data_x, nf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_y, h_data_y, nf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids_x, h_centroids_x, kf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids_y, h_centroids_y, kf, cudaMemcpyHostToDevice);

    int blocks_n = (sample_size + THREADS - 1) / THREADS;
    int blocks_k = (k + THREADS - 1) / THREADS;

    float* h_diff = new float[k];

    for (int iter = 0; iter < max_iterations; ++iter) {
        assign_labels<<<blocks_n, THREADS>>>(d_data_x, d_data_y, d_centroids_x, d_centroids_y, d_labels, sample_size, k);
        clear_centroids<<<blocks_k, THREADS>>>(d_sum_x, d_sum_y, d_count, k);
        accumulate_centroids<<<blocks_n, THREADS>>>(d_data_x, d_data_y, d_labels, d_sum_x, d_sum_y, d_count, sample_size);
        update_centroids<<<blocks_k, THREADS>>>(d_centroids_x, d_centroids_y, d_sum_x, d_sum_y, d_count, d_diff, k);

        cudaMemcpy(h_diff, d_diff, kf, cudaMemcpyDeviceToHost);
        float max_diff = 0.0f;
        for (int i = 0; i < k; ++i)
            if (h_diff[i] > max_diff) max_diff = h_diff[i];

        if (max_diff < THRESHOLD * THRESHOLD) break;
    }

    cudaMemcpy(h_centroids_x, d_centroids_x, kf, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids_y, d_centroids_y, kf, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels, d_labels, ni, cudaMemcpyDeviceToHost);

    delete[] h_diff;
    cudaFree(d_data_x);
    cudaFree(d_data_y);
    cudaFree(d_centroids_x);
    cudaFree(d_centroids_y);
    cudaFree(d_labels);
    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_count);
    cudaFree(d_diff);
}
