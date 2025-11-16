#include <torch/library.h>
#include <torch/all.h>
#include <cuda_runtime.h>
#include "sha256.cuh"

namespace aftune_torch {

at::Tensor sha256_cuda(const at::Tensor& a, int64_t chunk_size) {
  at::Tensor a_contig = a.contiguous();
  const void* a_ptr = a_contig.data_ptr();
  
  size_t tensor_size = a_contig.numel() * a_contig.element_size();
  
  size_t num_chunks = (tensor_size + chunk_size - 1) / chunk_size;
  
  auto options = torch::TensorOptions()
                  .dtype(torch::kUInt8)
                  .device(a_contig.device());
  at::Tensor result = at::empty({static_cast<int64_t>(num_chunks), 32}, options);
  BYTE* result_ptr = result.data_ptr<BYTE>();
  
  mcm_cuda_sha256_hash_batch(
    (BYTE*)a_ptr,
    chunk_size,
    result_ptr,
    num_chunks,
    tensor_size
  );
  
  return result;
}

TORCH_LIBRARY_IMPL(aftune_torch, CUDA, m) {
  m.impl("sha256", &sha256_cuda);
}
}
