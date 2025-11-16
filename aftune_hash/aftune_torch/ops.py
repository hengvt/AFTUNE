import torch
import hashlib
import struct
from blake3 import blake3

__all__ = ["sha256", "blaze3", "merkle_root", "flat_root"]

def sha256_cpu(a, chunk_size):
    a_contig = a.contiguous().cpu()
    if a_contig.numel() == 0:
        tensor_bytes = b''
        tensor_size = 0
    else:
        tensor_flat = a_contig.reshape(-1)
        tensor_view = tensor_flat.view(torch.uint8)
        tensor_bytes = tensor_view.numpy().tobytes()
        tensor_size = len(tensor_bytes)
    num_chunks = (tensor_size + chunk_size - 1) // chunk_size
    
    result = torch.empty(num_chunks, 32, dtype=torch.uint8)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, tensor_size)
        chunk = tensor_bytes[start:end]
        hash_bytes = hashlib.sha256(chunk).digest()
        result[i] = torch.frombuffer(bytearray(hash_bytes), dtype=torch.uint8)
    
    return result

def blaze3_cpu(a, chunk_size):
    hash_func = lambda data: blake3(data).digest()
    
    a_contig = a.contiguous().cpu()
    if a_contig.numel() == 0:
        tensor_bytes = b''
        tensor_size = 0
    else:
        tensor_flat = a_contig.reshape(-1)
        tensor_view = tensor_flat.view(torch.uint8)
        tensor_bytes = tensor_view.numpy().tobytes()
        tensor_size = len(tensor_bytes)
    num_chunks = (tensor_size + chunk_size - 1) // chunk_size
    
    result = torch.empty(num_chunks, 32, dtype=torch.uint8)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, tensor_size)
        chunk = tensor_bytes[start:end]
        hash_bytes = hash_func(chunk)
        result[i] = torch.frombuffer(bytearray(hash_bytes), dtype=torch.uint8)
    
    return result

def sha256(a, chunk_size=128):
    if a.is_cuda:
        return torch.ops.aftune_torch.sha256.default(a, chunk_size)
    else:
        return sha256_cpu(a, chunk_size)

@torch.library.register_fake("aftune_torch::sha256")
def _(a, chunk_size=128):
    torch._check(chunk_size > 0, lambda: f"chunk_size must be positive, got {chunk_size}")
    tensor_size = a.numel() * a.element_size()
    num_chunks = (tensor_size + chunk_size - 1) // chunk_size
    return torch.empty(num_chunks, 32, dtype=torch.uint8, device=a.device)

def blaze3(a, chunk_size=128):
    if a.is_cuda:
        return torch.ops.aftune_torch.blaze3.default(a, chunk_size)
    else:
        return blaze3_cpu(a, chunk_size)

@torch.library.register_fake("aftune_torch::blaze3")
def _(a, chunk_size=128):
    torch._check(chunk_size > 0, lambda: f"chunk_size must be positive, got {chunk_size}")
    tensor_size = a.numel() * a.element_size()
    num_chunks = (tensor_size + chunk_size - 1) // chunk_size
    return torch.empty(num_chunks, 32, dtype=torch.uint8, device=a.device)

def merkle_root_cpu(a, chunk_size, use_blake3):
    if use_blake3:
        chunk_hashes_tensor = blaze3(a, chunk_size)
    else:
        chunk_hashes_tensor = sha256(a, chunk_size)
    
    chunk_hashes_cpu = chunk_hashes_tensor.cpu()
    n = chunk_hashes_cpu.shape[0]
    chunk_hashes = chunk_hashes_cpu.numpy().tobytes()
    
    if use_blake3:
        hash_func = lambda data: blake3(data).digest()
    else:
        hash_func = lambda data: hashlib.sha256(data).digest()
    
    current_level = []
    for i in range(n):
        index_bytes = struct.pack('>Q', i)
        chunk_start = i * 32
        chunk_end = chunk_start + 32
        leaf_hash = hash_func(index_bytes + chunk_hashes[chunk_start:chunk_end])
        current_level.append(leaf_hash)
    
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            if i + 1 < len(current_level):
                right = current_level[i + 1]
                parent_hash = hash_func(left + right)
            else:
                parent_hash = left
            next_level.append(parent_hash)
        current_level = next_level
    
    return current_level[0]

def merkle_root(a, chunk_size=128, use_blake3=True, use_gpu=True):
    return merkle_root_cpu(a, chunk_size, use_blake3)

def flat_root(a, chunk_size=128, use_blake3=True):
    if use_blake3:
        chunk_hashes_tensor = blaze3(a, chunk_size)
        hash_func = lambda data: blake3(data).digest()
    else:
        chunk_hashes_tensor = sha256(a, chunk_size)
        hash_func = lambda data: hashlib.sha256(data).digest()
    return hash_func(chunk_hashes_tensor.cpu().numpy().tobytes())
