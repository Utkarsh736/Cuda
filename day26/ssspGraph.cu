#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF INT_MAX  // Representation of infinity

__global__ void sssp_kernel(int* d_nodes, int* d_edges, int* d_weights, int* d_dist, bool* d_updated, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_nodes && d_updated[tid]) {
        d_updated[tid] = false;
        int start = d_nodes[tid];
        int end = d_nodes[tid + 1];
        for (int i = start; i < end; i++) {
            int neighbor = d_edges[i];
            int weight = d_weights[i];
            if (d_dist[tid] != INF && d_dist[tid] + weight < d_dist[neighbor]) {
                d_dist[neighbor] = d_dist[tid] + weight;
                d_updated[neighbor] = true;
            }
        }
    }
}

void sssp(int* h_nodes, int* h_edges, int* h_weights, int num_nodes, int source) {
    // Allocate host memory
    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    bool* h_updated = (bool*)malloc(num_nodes * sizeof(bool));

    // Initialize host memory
    for (int i = 0; i < num_nodes; i++) {
        h_dist[i] = INF;
        h_updated[i] = false;
    }
    h_dist[source] = 0;
    h_updated[source] = true;

    // Allocate device memory
    int *d_nodes, *d_edges, *d_weights, *d_dist;
    bool* d_updated;
    cudaMalloc((void**)&d_nodes, (num_nodes + 1) * sizeof(int));
    cudaMalloc((void**)&d_edges, h_nodes[num_nodes] * sizeof(int));
    cudaMalloc((void**)&d_weights, h_nodes[num_nodes] * sizeof(int));
    cudaMalloc((void**)&d_dist, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_updated, num_nodes * sizeof(bool));

    // Copy data from host to device
    cudaMemcpy(d_nodes, h_nodes, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, h_edges, h_nodes[num_nodes] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, h_nodes[num_nodes] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, h_dist, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_updated, h_updated, num_nodes * sizeof(bool), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_nodes + threads_per_block - 1) / threads_per_block;

    // Run the kernel for (num_nodes - 1) iterations
    for (int i = 0; i < num_nodes - 1; i++) {
        sssp_kernel<<<blocks_per_grid, threads_per_block>>>(d_nodes, d_edges, d_weights, d_dist, d_updated, num_nodes);
        cudaDeviceSynchronize();
    }

    // Copy result from device to host
    cudaMemcpy(h_dist, d_dist, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the distance array
    for (int i = 0; i < num_nodes; i++) {
        if (h_dist[i] == INF) {
            printf("Node %d is unreachable\n", i);
        } else {
            printf("Shortest distance to node %d is %d\n", i, h_dist[i]);
        }
    }

    // Free device memory
    cudaFree(d_nodes);
    cudaFree(d_edges);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_updated);

    // Free host memory
    free(h_dist);
    free(h_updated);
}

int main() {
    // Example graph in CSR format
    int h_nodes[] = {0, 2, 5, 7, 9};  // Node pointers
    int h_edges[] = {1, 2, 0, 2, 3, 0, 1, 1, 2};  // Edges
    int h_weights[] = {1, 4, 1, 2, 6, 4, 2, 1, 3};  // Weights
    int num_nodes = 4;
    int source = 0;

    sssp(h_nodes, h_edges, h_weights, num_nodes, source);

    return 0;
}
