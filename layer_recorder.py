import torch
import os
import json
import io
import aftune_torch
import time
import random
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import subprocess
import zstandard as zstd

def set_deterministic(seed, enabled):
    if not enabled:
        return
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True)
    

class LayerRecorder:
    def __init__(self, save_dir="block_records",
                 steps_per_block=5,
                 enabled=True, async_save=True,
                 chunk_size=4096, use_blake3=True, learning_rate=None,
                 max_pending_tasks=64,
                 deduplicate_storage=True,
                 optimizer_type='sgd',
                 layers_per_block=1,
                 checkpoint_interval=1,
                 activation_interval=1,
                 module_structure_dir=None,
                 model_name=None,
                 compress_storage=True,
                 compression_level=3):
        self.save_dir = save_dir
        self.module_structure_dir = module_structure_dir
        self.model_name = model_name
        self.steps_per_block = steps_per_block
        self.layers_per_block = layers_per_block
        self.checkpoint_interval = checkpoint_interval
        self.activation_interval = activation_interval
        self.enabled = enabled
        self.async_save = async_save
        self.chunk_size = chunk_size
        self.use_blake3 = use_blake3
        self.learning_rate = learning_rate
        self.max_pending_tasks = max_pending_tasks
        self.deduplicate_storage = deduplicate_storage
        self.optimizer_type = optimizer_type
        self.compress_storage = compress_storage
        self.compression_level = compression_level
        self.records = {}
        self.current_step = 0
        self.optimizer = None
        self.tracked_layers = set()
        self.dataset_records = {}
        self.warmup_mode = False
        
        self.use_lora = False
        self.lora_modules = {}
        
        self.layer_order = []
        self.layer_order_finalized = False
        
        self.layer_to_block = {}
        self.block_to_layers = {}
        self.layer_blocks_finalized = False
        self.block_checkpoint_interval = {}
        self.time_stats = {
            'hash_input': 0.0,
            'hash_output': 0.0,
            'hash_input_grad': 0.0,
            'hash_output_grad': 0.0,
            'hash_parameters': 0.0,
            'hash_optimizer_state': 0.0,
            'copy_forward_activations': 0.0,
            'copy_backward_grads': 0.0,
            'copy_parameters': 0.0,
            'copy_optimizer_state': 0.0,
            'copy_dataset_input': 0.0,
            'save_parameters': 0.0,
            'save_activations': 0.0,
            'save_grads': 0.0,
            'save_optimizer_state': 0.0,
            'save_hashes': 0.0,
            'cuda_synchronize': 0.0,
            'backpressure_wait': 0.0,
            'async_submit': 0.0,
            'count_hash_input': 0,
            'count_hash_output': 0,
            'count_hash_input_grad': 0,
            'count_hash_output_grad': 0,
            'count_hash_parameters': 0,
            'count_hash_optimizer_state': 0,
            'count_save_blocks': 0,
            'count_async_submit': 0,
            'count_copy_forward': 0,
            'count_copy_backward': 0,
            'count_copy_parameters': 0,
            'count_copy_optimizer_state': 0,
            'count_copy_dataset': 0,
            'count_cuda_sync': 0,
            'count_backpressure': 0
        }
        self.executor = ThreadPoolExecutor(max_workers=16) if async_save else None
        self.pending_futures = []
        self.stats_lock = threading.Lock()
        self.pinned_buffer_pool = {}
        self.buffer_pool_lock = threading.Lock()
        self.max_buffers_per_shape = 128
        if enabled:
            os.makedirs(save_dir, exist_ok=True)
    
    def compress_file_in_subprocess(self, filepath):
        if not self.compress_storage:
            return
        
        compressed_path = filepath + '.zst'
        
        compress_script = f"""
import zstandard as zstd
import os

input_file = {repr(filepath)}
output_file = {repr(compressed_path)}
compression_level = {self.compression_level}

with open(input_file, 'rb') as f_in:
    data = f_in.read()

cctx = zstd.ZstdCompressor(level=compression_level)
compressed = cctx.compress(data)

with open(output_file, 'wb') as f_out:
    f_out.write(compressed)

os.remove(input_file)
"""
        
        subprocess.Popen(
            [sys.executable, '-c', compress_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def decompress_tensor(self, compressed_data):
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed_data)
        
        buffer = io.BytesIO(decompressed)
        tensor = torch.load(buffer, weights_only=False)
        return tensor
    
    def save_tensor_compressed(self, tensor, filepath):
        torch.save(tensor, filepath)
        
        if self.compress_storage:
            self.compress_file_in_subprocess(filepath)
    
    def load_tensor_compressed(self, filepath):
        compressed_path = filepath + '.zst'
        if os.path.exists(compressed_path):
            with open(compressed_path, 'rb') as f:
                compressed_data = f.read()
            return self.decompress_tensor(compressed_data)
        elif os.path.exists(filepath):
            return torch.load(filepath, map_location='cpu', weights_only=False)
        else:
            return None
    
    def get_pinned_buffer(self, shape, dtype):
        if self.pinned_buffer_pool is None:
            return torch.empty(shape, dtype=dtype, pin_memory=True, device='cpu')
        
        buffer_key = (tuple(shape), dtype)
        
        with self.buffer_pool_lock:
            if buffer_key not in self.pinned_buffer_pool:
                self.pinned_buffer_pool[buffer_key] = []
            
            buffer_list = self.pinned_buffer_pool[buffer_key]
            
            for buf in buffer_list:
                if sys.getrefcount(buf) <= 3:
                    return buf
            
            new_buffer = torch.empty(shape, dtype=dtype, pin_memory=True, device='cpu')
            
            if len(buffer_list) < self.max_buffers_per_shape:
                buffer_list.append(new_buffer)
            
            return new_buffer
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_warmup_mode(self, enabled):
        self.warmup_mode = enabled
    
    def clear_time_stats(self):
        with self.stats_lock:
            for key in self.time_stats:
                self.time_stats[key] = 0.0 if not key.startswith('count_') else 0
        
    def get_hash(self, tensor, category):
        assert(tensor.is_contiguous())
        if (torch.cuda.is_available()):
            torch.cuda.synchronize()
        
        start_time = time.time()
        root = aftune_torch.ops.flat_root(tensor, chunk_size=self.chunk_size, use_blake3=self.use_blake3)
        result = root.hex()
        elapsed = time.time() - start_time
        
        with self.stats_lock:
            if category in self.time_stats:
                self.time_stats[category] += elapsed
                count_key = f'count_{category}'
                if count_key in self.time_stats:
                    self.time_stats[count_key] += 1
        
        return result
    
    def create_forward_hook(self, layer_name: str, is_first_layer: bool = False, is_last_layer: bool = False):
        if not self.enabled:
            def empty_hook(module, input, output):
                pass
            return empty_hook
        
        def hook(module, input, output):
            if layer_name not in self.records:
                self.records[layer_name] = {}
            if self.current_step not in self.records[layer_name]:
                self.records[layer_name][self.current_step] = {}
            
            record = self.records[layer_name][self.current_step]
            
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input
                
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            
            if not self.layer_order_finalized:
                if layer_name not in self.layer_order:
                    self.layer_order.append(layer_name)
                
                if is_last_layer and self.current_step == 0:
                    self.layer_order_finalized = True
                    self.finalize_layer_blocks()
            
            layer_block_idx = self.layer_to_block.get(layer_name, 0) if self.layer_blocks_finalized else 0
            should_save_activation = (layer_block_idx % self.activation_interval == 0)
            
            if self.deduplicate_storage:
                is_block_first = self.is_block_first_layer(layer_name)
                is_block_last = self.is_block_last_layer(layer_name)
                is_global_first = self.is_first_layer(layer_name)
                
                if is_global_first:
                    input_hash = self.get_hash(input_tensor, 'hash_input')
                    record['input_hash'] = input_hash
                
                if is_block_last:
                    output_hash = self.get_hash(output_tensor, 'hash_output')
                    record['output_hash'] = output_hash
                
                if not is_block_first and not is_block_last:
                    record['is_internal_layer'] = True
            else:
                input_hash = self.get_hash(input_tensor, 'hash_input')
                record['input_hash'] = input_hash
                
                output_hash = self.get_hash(output_tensor, 'hash_output')
                record['output_hash'] = output_hash
            
            copy_start = time.time()
            
            if self.deduplicate_storage:
                is_block_first = self.is_block_first_layer(layer_name)
                is_block_last = self.is_block_last_layer(layer_name)
                is_global_first = self.is_first_layer(layer_name)
                
                if is_global_first and should_save_activation:
                    if self.async_save:
                        cpu_tensor = self.get_pinned_buffer(input_tensor.shape, input_tensor.dtype)
                        cpu_tensor.copy_(input_tensor, non_blocking=True)
                        record['input_async'] = cpu_tensor
                    else:
                        record['input'] = input_tensor.detach().cpu()
                
                if is_block_last and should_save_activation:
                    if self.async_save:
                        cpu_tensor = self.get_pinned_buffer(output_tensor.shape, output_tensor.dtype)
                        cpu_tensor.copy_(output_tensor, non_blocking=True)
                        record['output_async'] = cpu_tensor
                    else:
                        record['output'] = output_tensor.detach().cpu()
            else:
                if should_save_activation:
                    if self.async_save:
                        cpu_tensor = self.get_pinned_buffer(input_tensor.shape, input_tensor.dtype)
                        cpu_tensor.copy_(input_tensor, non_blocking=True)
                        record['input_async'] = cpu_tensor
                    else:
                        record['input'] = input_tensor.detach().cpu()
                
                if should_save_activation:
                    if self.async_save:
                        cpu_tensor = self.get_pinned_buffer(output_tensor.shape, output_tensor.dtype)
                        cpu_tensor.copy_(output_tensor, non_blocking=True)
                        record['output_async'] = cpu_tensor
                    else:
                        record['output'] = output_tensor.detach().cpu()
            
            copy_time = time.time() - copy_start
            with self.stats_lock:
                self.time_stats['copy_forward_activations'] += copy_time
                self.time_stats['count_copy_forward'] += 1
            
        return hook
    
    def record_checkpoint(self, layer_name: str, module):
        if not self.enabled:
            return
        
        step_in_block = self.current_step % self.steps_per_block
        should_record_weights = (step_in_block == 0)
        
        if not should_record_weights:
            return
        
        step_block_idx = self.current_step // self.steps_per_block
        layer_block_id = self.layer_to_block.get(layer_name, 0)
        block_checkpoint_interval = self.block_checkpoint_interval.get(layer_block_id, self.checkpoint_interval)
        should_save_parameters = (step_block_idx % block_checkpoint_interval == 0)
        
        if layer_name not in self.records:
            self.records[layer_name] = {}
        if self.current_step not in self.records[layer_name]:
            self.records[layer_name][self.current_step] = {}
        
        record = self.records[layer_name][self.current_step]
        
        if self.use_lora:
            params_to_record = {name: param for name, param in module.named_parameters() if param.requires_grad}
        else:
            params_to_record = {name: param for name, param in module.named_parameters()}
        
        record['parameter_hashes'] = {}
        for name, param in params_to_record.items():
            record['parameter_hashes'][name] = self.get_hash(param, 'hash_parameters')
        
        if self.optimizer is not None:
            record['optimizer_state_hashes'] = {}
            for name, param in params_to_record.items():
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            hash_key = f"{name}.{k}"
                            record['optimizer_state_hashes'][hash_key] = self.get_hash(v, 'hash_optimizer_state')
        
        if should_save_parameters:
            if self.async_save:
                param_copy_start = time.time()
                record['parameters_async'] = {}
                for name, param in params_to_record.items():
                    param_cpu = self.get_pinned_buffer(param.shape, param.dtype)
                    param_cpu.copy_(param, non_blocking=True)
                    record['parameters_async'][name] = param_cpu
                
                param_copy_time = time.time() - param_copy_start
                with self.stats_lock:
                    self.time_stats['copy_parameters'] += param_copy_time
                    self.time_stats['count_copy_parameters'] += 1
            else:
                record['parameters'] = {}
                param_copy_start = time.time()
                for name, param in params_to_record.items():
                    record['parameters'][name] = param.detach().cpu()
                param_copy_time = time.time() - param_copy_start
                with self.stats_lock:
                    self.time_stats['copy_parameters'] += param_copy_time
                    self.time_stats['count_copy_parameters'] += 1
            
            if self.optimizer is not None:
                if self.async_save:
                    opt_copy_start = time.time()
                    record['optimizer_state_async'] = {}
                    
                    for name, param in params_to_record.items():
                        if param in self.optimizer.state:
                            state = self.optimizer.state[param]
                            record['optimizer_state_async'][name] = {}
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    v_cpu = self.get_pinned_buffer(v.shape, v.dtype)
                                    v_cpu.copy_(v, non_blocking=True)
                                    record['optimizer_state_async'][name][k] = v_cpu
                                else:
                                    record['optimizer_state_async'][name][k] = v
                    
                    opt_copy_time = time.time() - opt_copy_start
                    with self.stats_lock:
                        self.time_stats['copy_optimizer_state'] += opt_copy_time
                        self.time_stats['count_copy_optimizer_state'] += 1
                else:
                    record['optimizer_state'] = {}
                    opt_copy_start = time.time()
                    for name, param in params_to_record.items():
                        if param in self.optimizer.state:
                            state = self.optimizer.state[param]
                            record['optimizer_state'][name] = {}
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    record['optimizer_state'][name][k] = v.detach().cpu()
                                else:
                                    record['optimizer_state'][name][k] = v
                    
                    opt_copy_time = time.time() - opt_copy_start
                    with self.stats_lock:
                        self.time_stats['copy_optimizer_state'] += opt_copy_time
                        self.time_stats['count_copy_optimizer_state'] += 1
        
        layer_block_idx = self.layer_to_block.get(layer_name, 0) if self.layer_blocks_finalized else 0
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        os.makedirs(block_dir, exist_ok=True)
        
        layer_dir = os.path.join(block_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        
        checkpoint_hash_file = os.path.join(layer_dir, 'checkpoint_hashes.json')
        checkpoint_hashes = {
            'has_checkpoint': should_save_parameters
        }
        
        if 'parameter_hashes' in record:
            checkpoint_hashes['parameter_hashes'] = record['parameter_hashes']
        if 'optimizer_state_hashes' in record:
            checkpoint_hashes['optimizer_state_hashes'] = record['optimizer_state_hashes']
        
        with open(checkpoint_hash_file, 'w') as f:
            json.dump(checkpoint_hashes, f, indent=2)
        
        self.save_layer_record(layer_name, self.current_step)
        if layer_name in self.records and self.current_step in self.records[layer_name]:
            del self.records[layer_name][self.current_step]
    
    def create_backward_hook(self, layer_name: str, is_last_layer: bool = False):
        if not self.enabled:
            def empty_hook(module, grad_input, grad_output):
                pass
            return empty_hook
        
        def hook(module, grad_input, grad_output):
            if layer_name not in self.records:
                self.records[layer_name] = {}
            if self.current_step not in self.records[layer_name]:
                self.records[layer_name][self.current_step] = {}
            
            record = self.records[layer_name][self.current_step]
            
            input_grad = None
            output_grad = None
            
            if isinstance(grad_input, tuple):
                if len(grad_input) > 0 and grad_input[0] is not None:
                    input_grad = grad_input[0]
            elif grad_input is not None:
                input_grad = grad_input
            
            if isinstance(grad_output, tuple):
                if len(grad_output) > 0 and grad_output[0] is not None:
                    output_grad = grad_output[0]
            elif grad_output is not None:
                output_grad = grad_output
            
            if input_grad is not None or output_grad is not None:
                layer_block_idx = self.layer_to_block.get(layer_name, 0) if self.layer_blocks_finalized else 0
                should_save_grad = (layer_block_idx % self.activation_interval == 0)
                
                if self.deduplicate_storage:
                    is_block_first = self.is_block_first_layer(layer_name)
                    is_global_last = self.is_last_layer(layer_name)
                    
                    if is_block_first and input_grad is not None:
                        record['input_grad_hash'] = self.get_hash(input_grad, 'hash_input_grad')
                    
                    if is_global_last and output_grad is not None:
                        record['output_grad_hash'] = self.get_hash(output_grad, 'hash_output_grad')
                else:
                    if input_grad is not None:
                        record['input_grad_hash'] = self.get_hash(input_grad, 'hash_input_grad')
                    
                    if output_grad is not None:
                        record['output_grad_hash'] = self.get_hash(output_grad, 'hash_output_grad')
                
                grad_cpu_copy_start = time.time()
                
                if self.deduplicate_storage:
                    is_block_first = self.is_block_first_layer(layer_name)
                    is_global_last = self.is_last_layer(layer_name)
                    
                    if is_block_first and input_grad is not None and should_save_grad:
                        if self.async_save:
                            input_grad_cpu = self.get_pinned_buffer(input_grad.shape, input_grad.dtype)
                            input_grad_cpu.copy_(input_grad, non_blocking=True)
                            record['input_grad_async'] = input_grad_cpu
                        else:
                            record['input_grad'] = input_grad.detach().cpu()
                    
                    if is_global_last and output_grad is not None and should_save_grad:
                        if self.async_save:
                            output_grad_cpu = self.get_pinned_buffer(output_grad.shape, output_grad.dtype)
                            output_grad_cpu.copy_(output_grad, non_blocking=True)
                            record['output_grad_async'] = output_grad_cpu
                        else:
                            record['output_grad'] = output_grad.detach().cpu()
                else:
                    if input_grad is not None and should_save_grad:
                        if self.async_save:
                            input_grad_cpu = self.get_pinned_buffer(input_grad.shape, input_grad.dtype)
                            input_grad_cpu.copy_(input_grad, non_blocking=True)
                            record['input_grad_async'] = input_grad_cpu
                        else:
                            record['input_grad'] = input_grad.detach().cpu()
                    
                    if output_grad is not None and should_save_grad:
                        if self.async_save:
                            output_grad_cpu = self.get_pinned_buffer(output_grad.shape, output_grad.dtype)
                            output_grad_cpu.copy_(output_grad, non_blocking=True)
                            record['output_grad_async'] = output_grad_cpu
                        else:
                            record['output_grad'] = output_grad.detach().cpu()
                
                grad_cpu_copy_time = time.time() - grad_cpu_copy_start
                with self.stats_lock:
                    self.time_stats['copy_backward_grads'] += grad_cpu_copy_time
                    self.time_stats['count_copy_backward'] += 1
            
            self.save_layer_record(layer_name, self.current_step, save_hashes=True)
            if layer_name in self.records and self.current_step in self.records[layer_name]:
                del self.records[layer_name][self.current_step]
            
        return hook
    
    def record_dataset_input(self, input_ids, labels, loss):
        if not self.enabled:
            return
        
        if self.current_step not in self.dataset_records:
            self.dataset_records[self.current_step] = {}
        
        record = self.dataset_records[self.current_step]
        record['loss'] = loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss
        
        if self.async_save:
            dataset_copy_start = time.time()
            input_ids_cpu = self.get_pinned_buffer(input_ids.shape, input_ids.dtype)
            labels_cpu = self.get_pinned_buffer(labels.shape, labels.dtype)
            input_ids_cpu.copy_(input_ids, non_blocking=True)
            labels_cpu.copy_(labels, non_blocking=True)
            record['input_ids_async'] = input_ids_cpu
            record['labels_async'] = labels_cpu
            dataset_copy_time = time.time() - dataset_copy_start
            with self.stats_lock:
                self.time_stats['copy_dataset_input'] += dataset_copy_time
                self.time_stats['count_copy_dataset'] += 1
            
            record['input_ids_hash'] = self.get_hash(input_ids, 'hash_input')
            record['labels_hash'] = self.get_hash(labels, 'hash_input')
        else:
            record['input_ids_hash'] = self.get_hash(input_ids, 'hash_input')
            record['labels_hash'] = self.get_hash(labels, 'hash_input')
            dataset_copy_start = time.time()
            record['input_ids'] = input_ids.detach().cpu()
            record['labels'] = labels.detach().cpu()
            dataset_copy_time = time.time() - dataset_copy_start
            with self.stats_lock:
                self.time_stats['copy_dataset_input'] += dataset_copy_time
                self.time_stats['count_copy_dataset'] += 1
        
    def save_module_structure(self, layer_name: str, module):
        if not self.enabled:
            return
        
        structure_dir = os.path.join(self.module_structure_dir, self.model_name, 'module_structures')
        os.makedirs(structure_dir, exist_ok=True)
        structure_path = os.path.join(structure_dir, f'{layer_name}.pt')
        
        if os.path.exists(structure_path):
            return
        
        original_device = None
        if next(module.parameters(), None) is not None:
            original_device = next(module.parameters()).device
        
        module.cpu()
        torch.save(module, structure_path)
        
        if original_device is not None and str(original_device) != 'cpu':
            module.to(original_device)
        
    
    def increment_step(self):
        if self.current_step in self.dataset_records:
            self.save_dataset_record(self.current_step)
            del self.dataset_records[self.current_step]
        
        self.current_step += 1
        self.tracked_layers.clear()
    
    def finalize_layer_blocks(self):
        if self.layer_blocks_finalized:
            return
        
        self.layer_blocks_finalized = True
        
        is_dinov2_giant = (self.model_name == 'dinov2_giant')
        if is_dinov2_giant and self.checkpoint_interval != 1 and len(self.layer_order) > 0 and self.layer_order[0] == 'embeddings':
            embedding_layer = self.layer_order[0]
            remaining_layers = self.layer_order[1:]
            
            self.layer_to_block[embedding_layer] = 0
            self.block_to_layers[0] = [embedding_layer]
            self.block_checkpoint_interval[0] = 1
            
            for idx, layer_name in enumerate(remaining_layers):
                layer_block_id = (idx // self.layers_per_block) + 1
                self.layer_to_block[layer_name] = layer_block_id
                
                if layer_block_id not in self.block_to_layers:
                    self.block_to_layers[layer_block_id] = []
                self.block_to_layers[layer_block_id].append(layer_name)
        else:
            for idx, layer_name in enumerate(self.layer_order):
                layer_block_id = idx // self.layers_per_block
                self.layer_to_block[layer_name] = layer_block_id
                
                if layer_block_id not in self.block_to_layers:
                    self.block_to_layers[layer_block_id] = []
                self.block_to_layers[layer_block_id].append(layer_name)
        
    
    def is_first_layer(self, layer_name: str):
        return self.layer_order and len(self.layer_order) > 0 and layer_name == self.layer_order[0]
    
    def is_last_layer(self, layer_name: str):
        return self.layer_order and len(self.layer_order) > 0 and layer_name == self.layer_order[-1]
    
    def get_prev_layer(self, layer_name: str):
        if not self.layer_order or layer_name not in self.layer_order:
            return None
        idx = self.layer_order.index(layer_name)
        if idx > 0:
            return self.layer_order[idx - 1]
        return None
    
    def is_block_first_layer(self, layer_name: str):
        if not self.layer_blocks_finalized or layer_name not in self.layer_to_block:
            return True
        block_id = self.layer_to_block[layer_name]
        block_layers = self.block_to_layers[block_id]
        return layer_name == block_layers[0]
    
    def is_block_last_layer(self, layer_name: str):
        if not self.layer_blocks_finalized or layer_name not in self.layer_to_block:
            return True
        block_id = self.layer_to_block[layer_name]
        block_layers = self.block_to_layers[block_id]
        return layer_name == block_layers[-1]
    
    def get_next_layer(self, layer_name: str):
        if not self.layer_order or layer_name not in self.layer_order:
            return None
        idx = self.layer_order.index(layer_name)
        if idx < len(self.layer_order) - 1:
            return self.layer_order[idx + 1]
        return None
    
    
    def save_dataset_record_sync(self, step: int, record: dict, record_time: bool = True):
        block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        
        block_dir = os.path.join(self.save_dir, f"dataset_block_{block_idx}")
        os.makedirs(block_dir, exist_ok=True)
        
        step_dir = os.path.join(block_dir, f"step_{step_in_block}")
        os.makedirs(step_dir, exist_ok=True)
        
        if 'input_ids_async' in record:
            input_ids = record['input_ids_async']
            labels = record['labels_async']
            sync_start = time.time() if record_time else None
            torch.cuda.synchronize()
            if record_time:
                with self.stats_lock:
                    self.time_stats['cuda_synchronize'] += time.time() - sync_start
                    self.time_stats['count_cuda_sync'] += 1
        else:
            input_ids = record.get('input_ids')
            labels = record.get('labels')
        
        input_ids_hash = record.get('input_ids_hash', '')
        labels_hash = record.get('labels_hash', '')
        
        if input_ids is not None:
            self.save_tensor_compressed(input_ids, os.path.join(step_dir, 'input_ids.pt'))
        if labels is not None:
            self.save_tensor_compressed(labels, os.path.join(step_dir, 'labels.pt'))
        
        output_logits = record.get('output_logits')
        output_logits_hash = record.get('output_logits_hash', '')
        if output_logits is not None:
            self.save_tensor_compressed(output_logits, os.path.join(step_dir, 'output_logits.pt'))
        
        hash_info = {
            'input_ids_hash': input_ids_hash,
            'labels_hash': labels_hash,
            'loss': record.get('loss')
        }
        if output_logits_hash:
            hash_info['output_logits_hash'] = output_logits_hash
        if 'label' in record:
            hash_info['label'] = record['label']
        with open(os.path.join(step_dir, 'dataset_info.json'), 'w') as f:
            json.dump(hash_info, f, indent=2)
    
    def save_dataset_record(self, step: int):
        if self.warmup_mode:
            return
        
        if step not in self.dataset_records:
            return
        
        record = self.dataset_records[step]
        
        if self.async_save:
            self.check_backpressure()
            
            submit_start = time.time()
            record_copy = {k: v for k, v in record.items()}
            future = self.executor.submit(self.save_dataset_record_sync, step, record_copy, True)
            self.pending_futures.append(future)
            with self.stats_lock:
                self.time_stats['async_submit'] += time.time() - submit_start
                self.time_stats['count_async_submit'] += 1
        else:
            self.save_dataset_record_sync(step, record, True)
    
    def save_layer_record_sync(self, layer_name: str, step: int, record: dict, record_time: bool = True, save_hashes: bool = True):
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        
        layer_block_idx = self.layer_to_block.get(layer_name, 0) if self.layer_blocks_finalized else 0
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        os.makedirs(block_dir, exist_ok=True)
        
        layer_dir = os.path.join(block_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        os.makedirs(step_dir, exist_ok=True)
        
        if 'input_async' in record or 'output_async' in record:
            input_data = record.get('input_async')
            output_data = record.get('output_async')
            sync_start = time.time() if record_time else None
            torch.cuda.synchronize()
            if record_time:
                with self.stats_lock:
                    self.time_stats['cuda_synchronize'] += time.time() - sync_start
                    self.time_stats['count_cuda_sync'] += 1
        else:
            input_data = record.get('input')
            output_data = record.get('output')
        
        input_hash = record.get('input_hash', '')
        output_hash = record.get('output_hash', '')
        
        save_start = time.time() if record_time else None
        if input_data is not None:
            self.save_tensor_compressed(input_data, os.path.join(step_dir, 'input.pt'))
        if output_data is not None:
            self.save_tensor_compressed(output_data, os.path.join(step_dir, 'output.pt'))
        if record_time:
            with self.stats_lock:
                self.time_stats['save_activations'] += time.time() - save_start
        
        if 'input_grad_async' in record or 'output_grad_async' in record:
            input_grad_data = record.get('input_grad_async')
            output_grad_data = record.get('output_grad_async')
            sync_start = time.time() if record_time else None
            torch.cuda.synchronize()
            if record_time:
                with self.stats_lock:
                    self.time_stats['cuda_synchronize'] += time.time() - sync_start
                    self.time_stats['count_cuda_sync'] += 1
        else:
            input_grad_data = record.get('input_grad')
            output_grad_data = record.get('output_grad')
        
        input_grad_hash = record.get('input_grad_hash', '')
        output_grad_hash = record.get('output_grad_hash', '')
        
        save_start = time.time() if record_time else None
        if input_grad_data is not None:
            self.save_tensor_compressed(input_grad_data, os.path.join(step_dir, 'input_grad.pt'))
        if output_grad_data is not None:
            self.save_tensor_compressed(output_grad_data, os.path.join(step_dir, 'output_grad.pt'))
        if record_time:
            with self.stats_lock:
                self.time_stats['save_grads'] += time.time() - save_start
        
        if step_in_block == 0:
            if 'parameters_async' in record:
                parameters = record['parameters_async']
                sync_start = time.time() if record_time else None
                torch.cuda.synchronize()
                if record_time:
                    with self.stats_lock:
                        self.time_stats['cuda_synchronize'] += time.time() - sync_start
                        self.time_stats['count_cuda_sync'] += 1
            else:
                parameters = record.get('parameters')
            
            parameter_hashes = record.get('parameter_hashes', {})
            
            def save_state_dict(state_dict, subdir, prefix, time_stat_key):
                save_start = time.time() if record_time else None
                if state_dict:
                    dir_path = os.path.join(layer_dir, subdir)
                    os.makedirs(dir_path, exist_ok=True)
                    name_mapping = {}
                    for idx, (name, data) in enumerate(state_dict.items()):
                        safe_name = f'{prefix}{idx}'
                        name_mapping[safe_name] = name
                        torch.save(data, os.path.join(dir_path, f'{safe_name}.pt'))
                    with open(os.path.join(dir_path, 'name_mapping.json'), 'w') as f:
                        json.dump(name_mapping, f, indent=2)
                if record_time:
                    with self.stats_lock:
                        self.time_stats[time_stat_key] += time.time() - save_start
            
            save_state_dict(parameters, 'parameters', 'param_', 'save_parameters')
            
            if 'optimizer_state_async' in record:
                optimizer_state = record['optimizer_state_async']
                sync_start = time.time() if record_time else None
                torch.cuda.synchronize()
                if record_time:
                    with self.stats_lock:
                        self.time_stats['cuda_synchronize'] += time.time() - sync_start
                        self.time_stats['count_cuda_sync'] += 1
            else:
                optimizer_state = record.get('optimizer_state')
            
            optimizer_state_hashes = record.get('optimizer_state_hashes', {})
            save_state_dict(optimizer_state, 'optimizer_state', 'opt_state_', 'save_optimizer_state')
        else:
            parameter_hashes = {}
            optimizer_state_hashes = {}
        
        if save_hashes:
            save_start = time.time() if record_time else None
            hash_file = os.path.join(step_dir, 'hashes.json')
            
            if os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    hash_info = json.load(f)
            else:
                hash_info = {}
            
            if input_hash:
                hash_info['input_hash'] = input_hash
            
            if output_hash:
                hash_info['output_hash'] = output_hash
            
            if output_grad_hash:
                hash_info['output_grad_hash'] = output_grad_hash
            if input_grad_hash:
                hash_info['input_grad_hash'] = input_grad_hash
            
            with open(hash_file, 'w') as f:
                json.dump(hash_info, f, indent=2)
            if record_time:
                with self.stats_lock:
                    self.time_stats['save_hashes'] += time.time() - save_start
                    self.time_stats['count_save_blocks'] += 1
    
    def save_layer_record(self, layer_name: str, step: int, save_hashes: bool = True):
        if self.warmup_mode:
            return
        
        if layer_name not in self.records or step not in self.records[layer_name]:
            return
        
        record = self.records[layer_name][step]
        
        if self.async_save:
            self.check_backpressure()
            
            submit_start = time.time()
            record_copy = {k: v for k, v in record.items()}
            future = self.executor.submit(self.save_layer_record_sync, layer_name, step, record_copy, True, save_hashes)
            self.pending_futures.append(future)
            with self.stats_lock:
                self.time_stats['async_submit'] += time.time() - submit_start
                self.time_stats['count_async_submit'] += 1
        else:
            self.save_layer_record_sync(layer_name, step, record, True, save_hashes)
    
    def print_time_stats(self):
        print("\nTime statistics:")
        
        copy_categories = [
            ('copy_forward_activations', 'Forward activation copy', 'count_copy_forward'),
            ('copy_backward_grads', 'Backward gradient copy', 'count_copy_backward'),
            ('copy_parameters', 'Parameter copy', 'count_copy_parameters'),
            ('copy_optimizer_state', 'Optimizer state copy', 'count_copy_optimizer_state'),
            ('copy_dataset_input', 'Dataset input copy', 'count_copy_dataset')
        ]
        
        total_copy_time = 0
        total_copy_count = 0
        has_copy_detail = False
        for key, label, count_key in copy_categories:
            time_val = self.time_stats.get(key, 0)
            count_val = self.time_stats.get(count_key, 0)
            total_copy_time += time_val
            total_copy_count += count_val
            if time_val > 0.001:
                has_copy_detail = True
        
        if has_copy_detail:
            print(f"  Tensor copy: {total_copy_time:.3f} seconds")
        
        hash_categories = [
            ('hash_input', 'Input Hash', 'count_hash_input'),
            ('hash_output', 'Output Hash', 'count_hash_output'),
            ('hash_input_grad', 'Input Grad Hash', 'count_hash_input_grad'),
            ('hash_output_grad', 'Output Grad Hash', 'count_hash_output_grad'),
            ('hash_parameters', 'Parameter Hash', 'count_hash_parameters'),
            ('hash_optimizer_state', 'Optimizer State Hash', 'count_hash_optimizer_state')
        ]
        
        total_hash_time = 0
        total_hash_count = 0
        for key, label, count_key in hash_categories:
            time_val = self.time_stats.get(key, 0)
            count_val = self.time_stats.get(count_key, 0)
            total_hash_time += time_val
            total_hash_count += count_val
        if total_hash_time > 0.001:
            print(f"  Hash computation: {total_hash_time:.3f} seconds ({total_hash_count} times)")
        
        async_submit_time = self.time_stats.get('async_submit', 0)
        async_submit_count = self.time_stats.get('count_async_submit', 0)
        backpressure_time = self.time_stats.get('backpressure_wait', 0)
        backpressure_count = self.time_stats.get('count_backpressure', 0)
        
        total_sync_wait = async_submit_time + backpressure_time
        if total_sync_wait > 0.001:
            print(f"  Synchronous wait: {total_sync_wait:.3f} seconds")
        
        save_categories = [
            ('save_activations', 'Activation save'),
            ('save_grads', 'Gradient save'),
            ('save_parameters', 'Parameter save'),
            ('save_optimizer_state', 'Optimizer state save'),
            ('save_hashes', 'Hash file save')
        ]
        
        total_save_time = 0
        has_save_detail = False
        for key, label in save_categories:
            time_val = self.time_stats.get(key, 0)
            total_save_time += time_val
            if time_val > 0.001:
                has_save_detail = True
        
        cuda_sync_time = self.time_stats.get('cuda_synchronize', 0)
        cuda_sync_count = self.time_stats.get('count_cuda_sync', 0)
        total_save_time += cuda_sync_time
        
        if has_save_detail or cuda_sync_time > 0.001:
            if self.async_save:
                print(f"  Background save: {total_save_time:.3f} seconds")
            else:
                print(f"  Disk save: {total_save_time:.3f} seconds")
        
        if self.async_save:
            total_blocking_time = total_copy_time + total_hash_time + total_sync_wait
        else:
            total_blocking_time = total_copy_time + total_hash_time + total_save_time
        
        print(f"  Total overhead: {total_blocking_time:.3f} seconds")
    
    def wait_all_tasks(self):
        if self.executor is not None and self.pending_futures:
            for future in self.pending_futures:
                future.result()
            self.pending_futures.clear()
    
    def save_final_model_parameters(self, model, tracked_modules):
        if not self.enabled:
            return
        final_dir = os.path.join(self.save_dir, 'final_model')
        os.makedirs(final_dir, exist_ok=True)
        for layer_name, module in tracked_modules.items():
            layer_dir = os.path.join(final_dir, layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            params_dir = os.path.join(layer_dir, 'parameters')
            os.makedirs(params_dir, exist_ok=True)
            name_mapping = {}
            param_hashes = {}
            
            params_to_save = {name: param for name, param in module.named_parameters() 
                            if not self.use_lora or param.requires_grad}
            
            for idx, (name, param) in enumerate(params_to_save.items()):
                safe_name = f'param_{idx}'
                name_mapping[safe_name] = name
                tensor_cpu = param.detach().cpu()
                torch.save(tensor_cpu, os.path.join(params_dir, f'{safe_name}.pt'))
                param_hashes[name] = self.get_hash(param, 'hash_parameters')
            with open(os.path.join(params_dir, 'name_mapping.json'), 'w') as f:
                json.dump(name_mapping, f, indent=2)
            with open(os.path.join(layer_dir, 'parameter_hashes.json'), 'w') as f:
                json.dump(param_hashes, f, indent=2)
    
    def clear_buffer_pool(self):
        if self.pinned_buffer_pool is not None:
            with self.buffer_pool_lock:
                for buffer_list in self.pinned_buffer_pool.values():
                    buffer_list.clear()
                self.pinned_buffer_pool.clear()
    
    def print_storage_stats(self):
        if not self.enabled or not os.path.exists(self.save_dir):
            return
        
        print("\nStorage space statistics:")
        
        storage_stats = {
            'activations': {'size': 0, 'count': 0},
            'grads': {'size': 0, 'count': 0},
            'parameters': {'size': 0, 'count': 0},
            'optimizer_state': {'size': 0, 'count': 0},
            'dataset': {'size': 0, 'count': 0},
            'hashes': {'size': 0, 'count': 0},
            'metadata': {'size': 0, 'count': 0},
            'other': {'size': 0, 'count': 0}
        }
        
        final_model_dir = os.path.join(self.save_dir, 'final_model')
        for root, dirs, files in os.walk(self.save_dir):
            if root.startswith(final_model_dir):
                continue
            for file in files:
                filepath = os.path.join(root, file)
                file_size = os.path.getsize(filepath)
                
                categorized = False
                
                if file in ['input.pt', 'output.pt'] or file in ['input.pt.zst', 'output.pt.zst']:
                    storage_stats['activations']['size'] += file_size
                    storage_stats['activations']['count'] += 1
                    categorized = True
                
                elif file in ['input_grad.pt', 'output_grad.pt'] or file in ['input_grad.pt.zst', 'output_grad.pt.zst']:
                    storage_stats['grads']['size'] += file_size
                    storage_stats['grads']['count'] += 1
                    categorized = True
                
                elif 'parameters' in root and (file.startswith('param_') and (file.endswith('.pt') or file.endswith('.pt.zst'))):
                    storage_stats['parameters']['size'] += file_size
                    storage_stats['parameters']['count'] += 1
                    categorized = True
                
                elif 'optimizer_state' in root and (file.startswith('opt_state_') and (file.endswith('.pt') or file.endswith('.pt.zst'))):
                    storage_stats['optimizer_state']['size'] += file_size
                    storage_stats['optimizer_state']['count'] += 1
                    categorized = True
                
                elif file in ['input_ids.pt', 'labels.pt', 'input_ids.pt.zst', 'labels.pt.zst', 'dataset_info.json']:
                    storage_stats['dataset']['size'] += file_size
                    storage_stats['dataset']['count'] += 1
                    categorized = True
                
                elif file == 'hashes.json':
                    storage_stats['hashes']['size'] += file_size
                    storage_stats['hashes']['count'] += 1
                    categorized = True
                
                elif file in ['metadata.json', 'name_mapping.json']:
                    storage_stats['metadata']['size'] += file_size
                    storage_stats['metadata']['count'] += 1
                    categorized = True
                
                if not categorized:
                    storage_stats['other']['size'] += file_size
                    storage_stats['other']['count'] += 1
        
        total_size = sum(info['size'] for info in storage_stats.values())
        
        print(f"  Total usage: {total_size / (1024**3):.2f} GB")
        
        categories_display = [
            ('activations', 'Activations'),
            ('grads', 'Gradients'),
            ('parameters', 'Parameters'),
            ('optimizer_state', 'Optimizer State'),
            ('dataset', 'Dataset'),
            ('hashes', 'Hash records'),
            ('metadata', 'Metadata'),
            ('other', 'Other')
        ]
        
        for key, label in categories_display:
            info = storage_stats[key]
            if info['count'] > 0:
                size_gb = info['size'] / (1024**3)
                size_mb = info['size'] / (1024**2)
                percentage = info['size'] / total_size * 100 if total_size > 0 else 0
                
                if size_gb >= 1:
                    print(f"  {label}: {size_gb:.2f} GB ({percentage:.1f}%)")
                else:
                    print(f"  {label}: {size_mb:.1f} MB ({percentage:.1f}%)")
    
    def check_backpressure(self):
        if self.executor is None or not self.pending_futures:
            return
        
        self.pending_futures = [f for f in self.pending_futures if not f.done()]
        
        while len(self.pending_futures) >= self.max_pending_tasks:
            wait_start = time.time()
            oldest_future = self.pending_futures.pop(0)
            oldest_future.result()
            with self.stats_lock:
                self.time_stats['backpressure_wait'] += time.time() - wait_start
                self.time_stats['count_backpressure'] += 1
    
    def save_all_records(self):
        if not self.enabled:
            return
        
        for layer_name in list(self.records.keys()):
            for step in list(self.records[layer_name].keys()):
                self.save_layer_record(layer_name, step)
        
        for step in list(self.dataset_records.keys()):
            self.save_dataset_record(step)
        
        self.wait_all_tasks()
        
        saved_layers = self.layer_order if self.layer_order else []
        
        metadata = {
            'layers': saved_layers,
            'layer_order': self.layer_order,
            'total_steps': self.current_step,
            'steps_per_block': self.steps_per_block,
            'layers_per_block': self.layers_per_block,
            'checkpoint_interval': self.checkpoint_interval,
            'activation_interval': self.activation_interval,
            'layer_to_block': self.layer_to_block,
            'block_to_layers': self.block_to_layers,
            'block_checkpoint_interval': self.block_checkpoint_interval,
            'enabled': self.enabled,
            'async_save': self.async_save,
            'chunk_size': self.chunk_size,
            'use_blake3': self.use_blake3,
            'learning_rate': self.learning_rate,
            'deduplicate_storage': self.deduplicate_storage,
            'optimizer_type': self.optimizer_type,
            'use_lora': self.use_lora,
            'module_structure_dir': self.module_structure_dir,
            'model_name': self.model_name,
            'compress_storage': self.compress_storage,
            'compression_level': self.compression_level
        }
        
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.executor is not None:
            self.executor.shutdown(wait=True)
    
    def load_dataset_record(self, step: int, device):
        block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        
        block_dir = os.path.join(self.save_dir, f"dataset_block_{block_idx}")
        if not os.path.exists(block_dir):
            raise FileNotFoundError(f"Dataset block not found: {block_dir}")
        
        step_dir = os.path.join(block_dir, f"step_{step_in_block}")
        if not os.path.exists(step_dir):
            raise FileNotFoundError(f"Dataset step not found: {step_dir}")
        
        record = {}
        
        input_ids_path = os.path.join(step_dir, 'input_ids.pt')
        if self.file_exists(input_ids_path):
            tensor = self.load_tensor_compressed(input_ids_path)
            record['input_ids'] = self.to_device(tensor, device)
        labels_path = os.path.join(step_dir, 'labels.pt')
        if self.file_exists(labels_path):
            tensor = self.load_tensor_compressed(labels_path)
            record['labels'] = self.to_device(tensor, device)
        output_logits_path = os.path.join(step_dir, 'output_logits.pt')
        if self.file_exists(output_logits_path):
            tensor = self.load_tensor_compressed(output_logits_path)
            record['output_logits'] = self.to_device(tensor, device)
        
        info_file = os.path.join(step_dir, 'dataset_info.json')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info = json.load(f)
                record.update(info)
        
        return record
    
    def ensure_metadata_loaded(self):
        if not self.layer_order:
            metadata = self.load_metadata()
            self.layer_order = metadata['layer_order']
            self.deduplicate_storage = metadata['deduplicate_storage']
            self.layer_to_block = metadata['layer_to_block']
            self.block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
            self.layer_blocks_finalized = True
    
    def file_exists(self, path):
        return os.path.exists(path) or os.path.exists(path + '.zst')
    
    def to_device(self, tensor, device):
        return tensor.to(device)
    
    def load_metadata(self):
        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_module_structure(self, layer_name: str, device):
        structure_dir = os.path.join(self.module_structure_dir, self.model_name, 'module_structures')
        structure_path = os.path.join(structure_dir, f'{layer_name}.pt')
        
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Structure file not found for {layer_name}: {structure_path}")
        
        module = torch.load(structure_path, map_location=device, weights_only=False)
        
        file_size_mb = os.path.getsize(structure_path) / 1024 / 1024
        
        return module
    
    def load_layer_record(self, layer_name: str, step: int, device):
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        
        layer_block_idx = self.layer_to_block.get(layer_name, 0) if self.layer_blocks_finalized else 0
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        if not os.path.exists(block_dir):
            raise FileNotFoundError(f"Block not found: {block_dir}")
        
        layer_dir = os.path.join(block_dir, layer_name)
        if not os.path.exists(layer_dir):
            raise FileNotFoundError(f"Layer directory not found: {layer_dir}")
        
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        if not os.path.exists(step_dir):
            raise FileNotFoundError(f"Step not found: {step_dir}")
        
        record = {}
        
        input_path = os.path.join(step_dir, 'input.pt')
        if self.file_exists(input_path):
            tensor = self.load_tensor_compressed(input_path)
            record['input'] = self.to_device(tensor, device)
        output_path = os.path.join(step_dir, 'output.pt')
        if self.file_exists(output_path):
            tensor = self.load_tensor_compressed(output_path)
            record['output'] = self.to_device(tensor, device)
        input_grad_path = os.path.join(step_dir, 'input_grad.pt')
        if self.file_exists(input_grad_path):
            tensor = self.load_tensor_compressed(input_grad_path)
            record['input_grad'] = self.to_device(tensor, device)
        output_grad_path = os.path.join(step_dir, 'output_grad.pt')
        if self.file_exists(output_grad_path):
            tensor = self.load_tensor_compressed(output_grad_path)
            record['output_grad'] = self.to_device(tensor, device)
        
        hash_file = os.path.join(step_dir, 'hashes.json')
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                hash_info = json.load(f)
            record.update(hash_info)
        
        if step_in_block == 0:
            params_dir = os.path.join(layer_dir, 'parameters')
            if os.path.exists(params_dir):
                record['parameters'] = {}
                mapping_file = os.path.join(params_dir, 'name_mapping.json')
                with open(mapping_file, 'r') as f:
                    name_mapping = json.load(f)
                for safe_name, original_name in name_mapping.items():
                    file_path = os.path.join(params_dir, f'{safe_name}.pt')
                    if os.path.exists(file_path):
                        record['parameters'][original_name] = torch.load(file_path, map_location=device, weights_only=False)
            
            opt_dir = os.path.join(layer_dir, 'optimizer_state')
            if os.path.exists(opt_dir):
                mapping_file = os.path.join(opt_dir, 'name_mapping.json')
                if os.path.exists(mapping_file):
                    record['optimizer_state'] = {}
                    with open(mapping_file, 'r') as f:
                        name_mapping = json.load(f)
                    for safe_name, original_name in name_mapping.items():
                        file_path = os.path.join(opt_dir, f'{safe_name}.pt')
                        if os.path.exists(file_path):
                            record['optimizer_state'][original_name] = torch.load(file_path, map_location=device, weights_only=False)
            
            checkpoint_hash_file = os.path.join(layer_dir, 'checkpoint_hashes.json')
            if os.path.exists(checkpoint_hash_file):
                with open(checkpoint_hash_file, 'r') as f:
                    checkpoint_hashes = json.load(f)
                    if 'parameter_hashes' in checkpoint_hashes:
                        record['parameter_hashes'] = checkpoint_hashes['parameter_hashes']
                    if 'optimizer_state_hashes' in checkpoint_hashes:
                        record['optimizer_state_hashes'] = checkpoint_hashes['optimizer_state_hashes']
        
        return record
    
    def load_layer_input(self, layer_name: str, step: int, device):
        self.ensure_metadata_loaded()
        
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        layer_block_idx = self.layer_to_block.get(layer_name, 0)
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        layer_dir = os.path.join(block_dir, layer_name)
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        input_path = os.path.join(step_dir, 'input.pt')
        
        if not self.deduplicate_storage or self.is_first_layer(layer_name):
            if self.file_exists(input_path):
                tensor = self.load_tensor_compressed(input_path)
                return self.to_device(tensor, device)
            else:
                raise FileNotFoundError(f"Input file not found: {input_path}")
        
        current_block_idx = self.layer_to_block.get(layer_name, 0)
        prev_block_idx = current_block_idx - 1
        prev_block_last_layer = self.block_to_layers[prev_block_idx][-1]
        
        return self.load_layer_output(prev_block_last_layer, step, device)
    
    def load_layer_output(self, layer_name: str, step: int, device):
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        layer_block_idx = self.layer_to_block.get(layer_name, 0)
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        layer_dir = os.path.join(block_dir, layer_name)
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        output_path = os.path.join(step_dir, 'output.pt')
        
        tensor = self.load_tensor_compressed(output_path)
        return self.to_device(tensor, device)
    
    def load_layer_output_grad(self, layer_name: str, step: int, device):
        self.ensure_metadata_loaded()
        
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        layer_block_idx = self.layer_to_block.get(layer_name, 0)
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        layer_dir = os.path.join(block_dir, layer_name)
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        output_grad_path = os.path.join(step_dir, 'output_grad.pt')
        
        if not self.deduplicate_storage or self.is_last_layer(layer_name):
            if self.file_exists(output_grad_path):
                tensor = self.load_tensor_compressed(output_grad_path)
                return self.to_device(tensor, device)
            else:
                raise FileNotFoundError(f"Output grad file not found: {output_grad_path}")
        
        next_layer_idx = self.layer_order.index(layer_name) + 1
        next_layer_name = self.layer_order[next_layer_idx]
        
        return self.load_layer_input_grad(next_layer_name, step, device)
    
    def load_layer_input_grad(self, layer_name: str, step: int, device):
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        layer_block_idx = self.layer_to_block.get(layer_name, 0)
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        layer_dir = os.path.join(block_dir, layer_name)
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        input_grad_path = os.path.join(step_dir, 'input_grad.pt')
        
        tensor = self.load_tensor_compressed(input_grad_path)
        return self.to_device(tensor, device)
