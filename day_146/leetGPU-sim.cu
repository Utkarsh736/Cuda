// boids_simulation.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define RADIUS 5.0f
#define RADIUS2 (RADIUS * RADIUS)

// CUDA kernel to compute next state of agents
__global__ void boids_kernel(const float* agents, float* agents_next, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi = agents[4 * i + 0];
    float yi = agents[4 * i + 1];
    float vxi = agents[4 * i + 2];
    float vyi = agents[4 * i + 3];

    float vx_sum = 0.0f, vy_sum = 0.0f;
    int count = 0;

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float xj = agents[4 * j + 0];
        float yj = agents[4 * j + 1];

        float dx = xj - xi;
        float dy = yj - yi;
        float dist2 = dx * dx + dy * dy;

        if (dist2 <= RADIUS2) {
            vx_sum += agents[4 * j + 2];
            vy_sum += agents[4 * j + 3];
            count++;
        }
    }

    float new_vx = vxi;
    float new_vy = vyi;

    if (count > 0) {
        float avg_vx = vx_sum / count;
        float avg_vy = vy_sum / count;

        new_vx = 0.5f * vxi + 0.5f * avg_vx;
        new_vy = 0.5f * vyi + 0.5f * avg_vy;
    }

    float new_x = xi + new_vx;
    float new_y = yi + new_vy;

    agents_next[4 * i + 0] = new_x;
    agents_next[4 * i + 1] = new_y;
    agents_next[4 * i + 2] = new_vx;
    agents_next[4 * i + 3] = new_vy;
}

extern "C" void solve(const float* h_agents, float* h_agents_next, int N) {
    float *d_agents, *d_agents_next;
    size_t sz = 4 * N * sizeof(float);

    cudaMalloc(&d_agents, sz);
    cudaMalloc(&d_agents_next, sz);

    cudaMemcpy(d_agents, h_agents, sz, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    boids_kernel<<<blocks, threads>>>(d_agents, d_agents_next, N);

    cudaMemcpy(h_agents_next, d_agents_next, sz, cudaMemcpyDeviceToHost);

    cudaFree(d_agents);
    cudaFree(d_agents_next);
}
