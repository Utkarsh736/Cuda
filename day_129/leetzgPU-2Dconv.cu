#include <cuda_runtime.h>
#include <iostream>

:contentReference[oaicite:1]{index=1}

__global__
:contentReference[oaicite:2]{index=2}
                  :contentReference[oaicite:3]{index=3}
                  :contentReference[oaicite:4]{index=4}
{
    :contentReference[oaicite:5]{index=5}
    :contentReference[oaicite:6]{index=6}
    :contentReference[oaicite:7]{index=7}

    __shared__ float tile[TILE_W][TILE_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int out_row = blockIdx.y * O_TILE_H + ty;
    int out_col = blockIdx.x * O_TILE_W + tx;

    int row_start = out_row;
    int col_start = out_col;

    // Load tile (including halo for convolution)
    if ((row_start < in_rows) && (col_start < in_cols))
        tile[ty][tx] = input[row_start * in_cols + col_start];
    else
        tile[ty][tx] = 0.0f;

    __syncthreads();

    // Compute only valid output elements
    if (ty < O_TILE_H && tx < O_TILE_W
        && out_row <= in_rows - k_rows
        && out_col <= in_cols - k_cols)
    {
        float sum = 0.0f;
        for (int r = 0; r < k_rows; ++r)
            for (int c = 0; c < k_cols; ++c)
                sum += tile[ty + r][tx + c] * d_kernel[r * k_cols + c];
        output[out_row * (in_cols - k_cols + 1) + out_col] = sum;
    }
}

:contentReference[oaicite:8]{index=8}
           :contentReference[oaicite:9]{index=9}
{
    :contentReference[oaicite:10]{index=10}
    :contentReference[oaicite:11]{index=11}
    :contentReference[oaicite:12]{index=12}
    :contentReference[oaicite:13]{index=13}

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_sz);
    cudaMemcpy(d_in, input, in_sz, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, out_sz);

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, kernel,
                       k_rows * k_cols * sizeof(float));

    dim3 threads(TILE_W, TILE_W);
    dim3 blocks((out_cols + O_TILE_W - 1) / O_TILE_W,
                (out_rows + O_TILE_H - 1) / O_TILE_H);

    size_t shared = TILE_W * TILE_W * sizeof(float);
    conv2d_tiled<<<blocks, threads, shared>>>(d_in, d_out,
                                              in_rows, in_cols,
                                              k_rows, k_cols);

    cudaMemcpy(output, d_out, out_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
