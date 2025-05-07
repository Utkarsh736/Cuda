# sort_kernel.py

import torch
from torch.utils.cpp_extension import load_inline

src = """
#include <torch/extension.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

torch::Tensor sort_cuda(torch::Tensor input) {
    auto input_contig = input.contiguous();
    auto output = input_contig.clone();
    auto N = output.numel();

    // Thrust device pointer
    thrust::device_ptr<float> begin(output.data_ptr<float>());
    thrust::device_ptr<float> end = begin + N;

    // In-place sort (ascending)
    thrust::sort(thrust::device, begin, end);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sort_cuda", &sort_cuda, "CUDA sort using Thrust");
}
"""

sort_module = load_inline(name="sort_cuda_ext", cpp_sources=src,
                          functions=["sort_cuda"],
                          extra_cflags=["-O3"],
                          extra_cuda_cflags=["-O3"],
                          with_cuda=True)

def sort(input_tensor: torch.Tensor) -> torch.Tensor:
    if not input_tensor.is_cuda or input_tensor.dtype != torch.float32:
        raise ValueError("Input must be a CUDA float32 tensor")
    return sort_module.sort_cuda(input_tensor)