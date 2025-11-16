#include <torch/library.h>
#include <torch/all.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "blaze3.cuh"

namespace aftune_torch {

__device__ void g_merge_chunks(Chunk &left, Chunk &right, Chunk &parent) {
    left.g_compress_chunk();
    right.g_compress_chunk();
    
    parent.flags = left.flags | PARENT;
    g_memcpy(parent.key, left.key, 32);
    parent.counter = 0;
    parent.leaf_len = 0;
    
    g_memcpy(parent.data, left.raw_hash, 32);
    g_memcpy(parent.data + 8, right.raw_hash, 32);
}

__global__ void blaze3_chunks_kernel(
    const char* input_data,
    size_t total_bytes,
    size_t chunk_size,
    size_t num_chunks,
    u8* output_hashes,
    u32* key
) {
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (chunk_idx >= num_chunks) return;
    
    size_t chunk_start = chunk_idx * chunk_size;
    size_t chunk_end = min(chunk_start + chunk_size, total_bytes);
    size_t actual_chunk_size = chunk_end - chunk_start;
    
    if (actual_chunk_size <= CHUNK_LEN) {
        Chunk chunk;
        chunk.init_leaf((char*)input_data + chunk_start, actual_chunk_size, 0, key, 0);
        chunk.g_compress_chunk(ROOT);
        
        u8* output_ptr = output_hashes + chunk_idx * 32;
        u32* raw_hash = chunk.raw_hash;
        
        for (int i = 0; i < 8; i++) {
            u32 word = raw_hash[i];
            output_ptr[i * 4 + 0] = (word >> 0) & 0xFF;
            output_ptr[i * 4 + 1] = (word >> 8) & 0xFF;
            output_ptr[i * 4 + 2] = (word >> 16) & 0xFF;
            output_ptr[i * 4 + 3] = (word >> 24) & 0xFF;
        }
        return;
    }
    
    size_t num_blake3_chunks = (actual_chunk_size + CHUNK_LEN - 1) / CHUNK_LEN;
    const int MAX_SUBCHUNKS = 64;
    
    if (num_blake3_chunks > MAX_SUBCHUNKS) {
        Chunk chunk;
        chunk.init_leaf((char*)input_data + chunk_start, CHUNK_LEN, 0, key, 0);
        chunk.g_compress_chunk(ROOT);
        
        u8* output_ptr = output_hashes + chunk_idx * 32;
        u32* raw_hash = chunk.raw_hash;
        for (int i = 0; i < 8; i++) {
            u32 word = raw_hash[i];
            output_ptr[i * 4 + 0] = (word >> 0) & 0xFF;
            output_ptr[i * 4 + 1] = (word >> 8) & 0xFF;
            output_ptr[i * 4 + 2] = (word >> 16) & 0xFF;
            output_ptr[i * 4 + 3] = (word >> 24) & 0xFF;
        }
        return;
    }
    
    Chunk sub_chunks[MAX_SUBCHUNKS];
    for (int i = 0; i < num_blake3_chunks; i++) {
        size_t sub_start = chunk_start + i * CHUNK_LEN;
        size_t sub_size = min((size_t)CHUNK_LEN, chunk_end - sub_start);
        sub_chunks[i].init_leaf((char*)input_data + sub_start, sub_size, 0, key, i);
        sub_chunks[i].g_compress_chunk();
    }
    
    int current_level_size = num_blake3_chunks;
    Chunk temp_chunks[MAX_SUBCHUNKS];
    bool use_temp = false;
    
    while (current_level_size > 1) {
        int next_level_size = (current_level_size + 1) / 2;
        Chunk* src = use_temp ? temp_chunks : sub_chunks;
        Chunk* dst = use_temp ? sub_chunks : temp_chunks;
        
        for (int i = 0; i < current_level_size / 2; i++) {
            g_merge_chunks(src[i * 2], src[i * 2 + 1], dst[i]);
        }
        
        if (current_level_size % 2 == 1) {
            dst[next_level_size - 1] = src[current_level_size - 1];
        }
        
        current_level_size = next_level_size;
        use_temp = !use_temp;
    }
    
    Chunk* final_chunk = use_temp ? temp_chunks : sub_chunks;
    final_chunk[0].g_compress_chunk(ROOT);
    
    u8* output_ptr = output_hashes + chunk_idx * 32;
    u32* raw_hash = final_chunk[0].raw_hash;
    
    for (int i = 0; i < 8; i++) {
        u32 word = raw_hash[i];
        output_ptr[i * 4 + 0] = (word >> 0) & 0xFF;
        output_ptr[i * 4 + 1] = (word >> 8) & 0xFF;
        output_ptr[i * 4 + 2] = (word >> 16) & 0xFF;
        output_ptr[i * 4 + 3] = (word >> 24) & 0xFF;
    }
}

at::Tensor blaze3_cuda(const at::Tensor& a, int64_t chunk_size) {
    at::Tensor a_contig = a.contiguous();
    const void* a_ptr = a_contig.data_ptr();

    size_t tensor_size = a_contig.numel() * a_contig.element_size();
    
    size_t num_chunks = (tensor_size + chunk_size - 1) / chunk_size;
    
    auto options = torch::TensorOptions()
                    .dtype(torch::kUInt8)
                    .device(a_contig.device());
    at::Tensor output = torch::empty({static_cast<int64_t>(num_chunks), 32}, options);
    u8* output_ptr = output.data_ptr<u8>();
    
    u32* d_key;
    cudaMalloc(&d_key, 8 * sizeof(u32));
    cudaMemcpy(d_key, IV, 8 * sizeof(u32), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int num_blocks = (num_chunks + threads_per_block - 1) / threads_per_block;
    
    blaze3_chunks_kernel<<<num_blocks, threads_per_block>>>(
        (const char*)a_ptr,
        tensor_size,
        chunk_size,
        num_chunks,
        output_ptr,
        d_key
    );
    
    cudaDeviceSynchronize();
    
    cudaFree(d_key);

    return output;
}

TORCH_LIBRARY_IMPL(aftune_torch, CUDA, m) {
    m.impl("blaze3", &blaze3_cuda);
}
}
