//!POPCORN leaderboard conv2d
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE_X 16         // tile width
#define TILE_Y 16         // tile height

// 2D convolution with shared memory, vectorized loads, and unrolling
__global__ void conv2d_optimized(const float* __restrict__ input,
                                 const float* __restrict__ kernel,
                                 float* __restrict__ output,
                                 int B, int C, int H, int W, int K) {
    // Shared memory for input tile + halo and kernel
    extern __shared__ float sdata[];
    float* s_input = sdata;                                           // size (TILE_Y+K-1)*(TILE_X+K-1)
    float* s_kernel = s_input + (TILE_Y + K - 1)*(TILE_X + K - 1);     // size C*K*K

    // Thread indices
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * TILE_X, by = blockIdx.y * TILE_Y, bc = blockIdx.z;  // bc = output channel+batch

    int b = bc / C, m = bc % C;       // batch index and output channel
    int out_y = by + ty, out_x = bx + tx;

    // Load kernel for this output channel into shared memory (once per block)
    for (int c = 0; c < C; ++c) {
        for (int i = ty; i < K; i += blockDim.y) {
            for (int j = tx; j < K; j += blockDim.x) {
                int idx = ((m*C + c)*K + i)*K + j;
                s_kernel[(c*K + i)*K + j] = kernel[idx];
            }
        }
    }
    __syncthreads();  // Ensure kernel loaded 4

    // Load input tile (with halo) into shared memory via float4 vector loads
    int halo = K - 1;
    int shared_width = TILE_X + halo;
    int shared_height= TILE_Y + halo;

    int global_y = by + ty - halo/2;
    for (int y = ty; y < shared_height; y += blockDim.y) {
        int in_y = by + y - halo/2;
        bool y_in = (in_y >= 0 && in_y < H);
        for (int x = tx; x < shared_width; x += blockDim.x) {
            int in_x = bx + x - halo/2;
            bool x_in = (in_x >= 0 && in_x < W);
            int idx = y * shared_width + x;
            if (y_in && x_in) {
                // Vectorized load
                auto ptr = reinterpret_cast<const float4*>(input + ((b*C + 0)*H + in_y)*W + in_x);
                float4 v = ptr[0];  // single float4 load 5
                // store components (assuming TILE_X aligned to 4)
                int base = idx*4;
                ((float4*)(s_input + base))[0] = v;
            } else {
                // zero‐pad out‐of‐bounds
                int base = idx*4;
                ((float4*)(s_input + base))[0] = make_float4(0,0,0,0);
            }
        }
    }
    __syncthreads();  // Ensure input tile loaded 6

    // Only threads that map to valid output pixels compute
    if (out_x < W - K + 1 && out_y < H - K + 1) {
        float sum = 0.0f;
        // Unroll over channels and kernel
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < K; ++i) {
                #pragma unroll
                for (int j = 0; j < K; ++j) {
                    int sy = ty + i;
                    int sx = tx + j;
                    float in_val = s_input[(sy*shared_width + sx)];
                    float w = s_kernel[(c*K + i)*K + j];
                    sum += in_val * w;
                }
            }
        }
        // Write to global memory
        int out_idx = ((b*C + m)*(H - K + 1) + out_y)*(W - K + 1) + out_x;
        output[out_idx] = sum;
    }
}

// Host‐side launcher
void custom_kernel(const torch::Tensor& input_tensor,
                   const torch::Tensor& kernel_tensor,
                   torch::Tensor& output_tensor) {
    int B = input_tensor.size(0), C = input_tensor.size(1);
    int H = input_tensor.size(2), W = input_tensor.size(3);
    int K = kernel_tensor.size(2);

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((W-K+1 + TILE_X-1)/TILE_X,
              (H-K+1 + TILE_Y-1)/TILE_Y,
              B*C);

    // Shared memory: input tile + kernel
    size_t smem =  sizeof(float)*((TILE_Y+K-1)*(TILE_X+K-1)*4  // float4 in s_input
                 + C*K*K);                                       // s_kernel

    conv2d_optimized<<<grid, block, smem>>>(
        input_tensor.data_ptr<float>(),
        kernel_tensor.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        B, C, H, W, K
    );
    cudaDeviceSynchronize();  // ensure completion
}