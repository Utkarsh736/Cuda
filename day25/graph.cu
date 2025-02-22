#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define INF 2147483647  // Representation of infinity

__global__ void bfs_kernel(int* d_nodes, int* d_edges, int* d_cost, int* d_frontier, int* d_next_frontier, int* d_frontier_size, int* d_next_frontier_size, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *d_frontier_size) {
        int node = d_frontier[tid];
        int start = d_nodes[node];
        int end = d_nodes[node + 1];
        for (int i = start; i < end; i++) {
            int neighbor = d_edges[i];
            if (d_cost[neighbor] == INF) {
                d_cost[neighbor] = level + 1;
                int index = atomicAdd(d_next_frontier_size, 1);
                d_next_frontier[index] = neighbor;
            }
        }
    }
}

void bfs(int* h_nodes, int* h_edges, int num_nodes, int source) {
    // Allocate host memory
    int* h_cost = (int*)malloc(num_nodes * sizeof(int));
    int* h_frontier = (int*)malloc(num_nodes * sizeof(int));
    int* h_next_frontier = (int*)malloc(num_nodes * sizeof(int));
    int h_frontier_size = 1;
    int h_next_frontier_size = 0;

    // Initialize host memory
    for (int i = 0; i < num_nodes; i++) {
        h_cost[i] = INF;
    }
    h_cost[source] = 0;
    h_frontier[0] = source;

    // Allocate device memory
    int *d_nodes, *d_edges, *d_cost, *d_frontier, *d_next_frontier, *d_frontier_size, *d_next_frontier_size;
    cudaMalloc((void**)&d_nodes, (num_nodes + 1) * sizeof(int));
    cudaMalloc((void**)&d_edges, h_nodes[num_nodes] * sizeof(int));
    cudaMalloc((void**)&d_cost, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_frontier, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_next_frontier, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_frontier_size, sizeof(int));
    cudaMalloc((void**)&d_next_frontier_size, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_nodes, h_nodes, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, h_edges, h_nodes[num_nodes] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, h_cost, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, h_frontier, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_size, &h_frontier_size, sizeof(int), cudaMemcpyHostToDevice);

    int level = 0;
    while (h_frontier_size > 0) {
        int threads_per_block = 256;
        int blocks_per_grid = (h_frontier_size + threads_per_block - 1) / threads_per_block;

        // Initialize next frontier size on device
        cudaMemcpy(d_next_frontier_size, &h_next_frontier_size, sizeof(int), cudaMemcpyHostToDevice);

        // Launch BFS kernel
        bfs_kernel<<<blocks_per_grid, threads_per_block>>>(d_nodes, d_edges, d_cost, d_frontier, d_next_frontier, d_frontier_size, d_next_frontier_size, level);
        cudaDeviceSynchronize();

        // Copy next frontier size from device to host
        cudaMemcpy(&h_next_frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap frontiers
        int* temp = d_frontier;
        d_frontier = d_next_frontier;
        d_next_frontier = temp;

        h_frontier_size = h_next_frontier_size;
        h_next_frontier_size = 0;
        level++;

        // Copy new frontier size to device
        cudaMemcpy(d_frontier_size, &h_frontier_size, sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy result from device to host
    cudaMemcpy(h_cost, d_cost, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the cost array
    for (int i = 0; i < num_nodes; i++) {
        if (h_cost[i] == INF) {
            printf("Node %d is unreachable\n", i);
        } else {
            printf("Cost to reach node %d is %d\n", i, h_cost[i]);
        }
    }

    // Free device memory
    cudaFree(d_nodes);
    cudaFree(d_edges);
    cudaFree(d_cost);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_size);
    cudaFree(d_next_frontier_size);

    // Free host memory
    free(h_cost);
    free(h_frontier);
    free(h_next_frontier);
}

int main() {
    // Example graph in CSR format
    int h_nodes[] = {0, 2, 5, 7, 9};  // Node pointers
    int h_edges[] = {1, 2, 0, 2, 3, 0, 1, 1, 2};  // Edges
    int num_nodes = 4;
    int source = 0;

    bfs(h_nodes, h_edges, num_nodes, source);

    return 0;
}
