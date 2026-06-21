import torch
import os
import json
import io
import warnings
import aftune_torch
import time
import random
import sys
import numpy as np

# Expected when module inputs do not require grad (e.g. embedding + input_ids)
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing when gradients are computed with respect to module outputs",
    category=UserWarning,
)
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
    

# Records per-layer activations, grads, weights, and optimizer state to disk
class Recorder:
    def __init__(self, save_dir="block_records",
                 steps_per_block=1,
                 layers_per_block=1,
                 enabled=True, async_save=True,
                 chunk_size=4096, use_blake3=True, learning_rate=None,
                 max_pending_tasks=64,
                 optimizer_type='sgd',
                 checkpoint_interval=1,
                 activation_interval=1,
                 module_structure_dir=None,
                 model_name=None,
                 compress_storage=True,
                 compression_level=3,
                 device=None):
        self.save_dir = save_dir
        self.device = device
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
        }
        self.executor = ThreadPoolExecutor(max_workers=16) if async_save else None
        self.pending_futures = []
        self.compress_processes = []
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
        
        process = subprocess.Popen(
            [sys.executable, '-c', compress_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.compress_processes.append(process)
    
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

    def copy_tensor_to_record(self, tensor, record, sync_key, async_key):
        if self.async_save:
            cpu_tensor = self.get_pinned_buffer(tensor.shape, tensor.dtype)
            cpu_tensor.copy_(tensor, non_blocking=True)
            record[async_key] = cpu_tensor
        else:
            record[sync_key] = tensor.detach().cpu()

    def copy_named_tensors_to_record(self, tensors, record, sync_key, async_key, time_key):
        copy_start = time.time()
        if self.async_save:
            record[async_key] = {}
            for name, tensor in tensors.items():
                cpu_tensor = self.get_pinned_buffer(tensor.shape, tensor.dtype)
                cpu_tensor.copy_(tensor, non_blocking=True)
                record[async_key][name] = cpu_tensor
        else:
            record[sync_key] = {name: tensor.detach().cpu() for name, tensor in tensors.items()}
        with self.stats_lock:
            self.time_stats[time_key] += time.time() - copy_start

    def copy_optimizer_state_to_record(self, params_to_record, record):
        copy_start = time.time()
        if self.async_save:
            record['optimizer_state_async'] = {}
            for name, param in params_to_record.items():
                if param not in self.optimizer.state:
                    continue
                state = self.optimizer.state[param]
                record['optimizer_state_async'][name] = {}
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        v_cpu = self.get_pinned_buffer(v.shape, v.dtype)
                        v_cpu.copy_(v, non_blocking=True)
                        record['optimizer_state_async'][name][k] = v_cpu
                    else:
                        record['optimizer_state_async'][name][k] = v
        else:
            record['optimizer_state'] = {}
            for name, param in params_to_record.items():
                if param not in self.optimizer.state:
                    continue
                state = self.optimizer.state[param]
                record['optimizer_state'][name] = {}
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        record['optimizer_state'][name][k] = v.detach().cpu()
                    else:
                        record['optimizer_state'][name][k] = v
        with self.stats_lock:
            self.time_stats['copy_optimizer_state'] += time.time() - copy_start

    def enqueue_async_save(self, record, save_sync):
        if not self.async_save:
            save_sync(record)
            return
        self.check_backpressure()
        submit_start = time.time()
        record_copy = {k: v for k, v in record.items()}
        future = self.executor.submit(save_sync, record_copy)
        self.pending_futures.append(future)
        with self.stats_lock:
            self.time_stats['async_submit'] += time.time() - submit_start

    def cuda_sync_for_save(self, record_time):
        if not record_time:
            torch.cuda.synchronize(self.device)
            return
        sync_start = time.time()
        torch.cuda.synchronize(self.device)
        with self.stats_lock:
            self.time_stats['cuda_synchronize'] += time.time() - sync_start

    def storage_file_category(self, file, root):
        if file in ('input.pt', 'output.pt', 'input.pt.zst', 'output.pt.zst'):
            return 'activations'
        if file in ('input_grad.pt', 'output_grad.pt', 'input_grad.pt.zst', 'output_grad.pt.zst'):
            return 'grads'
        if 'parameters' in root and file.startswith('param_') and file.endswith(('.pt', '.pt.zst')):
            return 'parameters'
        if 'optimizer_state' in root and file.startswith('opt_state_') and file.endswith(('.pt', '.pt.zst')):
            return 'optimizer_state'
        if file == 'hashes.json':
            return 'hashes'
        return 'other'
        
    # Map-reduce hash via aftune_torch.ops.flat_root
    def get_hash(self, tensor, category):
        assert(tensor.is_contiguous())
        #if tensor.is_cuda:
        #    torch.cuda.synchronize(tensor.device)
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
    
    # Forward hook: hash and optionally copy block boundary activations
    def create_forward_hook(self, layer_name: str, is_last_layer: bool = False):
        if not self.enabled:
            def empty_hook(module, input, output):
                pass
            return empty_hook
        
        def hook(module, input, output):
            if not self.layer_order_finalized:
                if layer_name not in self.layer_order:
                    self.layer_order.append(layer_name)
                if is_last_layer and self.current_step == 0:
                    self.layer_order_finalized = True
                    self.finalize_layer_blocks()
            
            if self.layer_blocks_finalized and not self.is_block_first_layer(layer_name) and not self.is_block_last_layer(layer_name):
                return
            
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
            
            layer_block_idx = self.layer_to_block.get(layer_name, 0) if self.layer_blocks_finalized else 0
            should_save_activation = (layer_block_idx % self.activation_interval == 0)
            
            is_block_last = self.is_block_last_layer(layer_name)
            is_global_first = self.is_first_layer(layer_name)
            if is_global_first:
                record['input_hash'] = self.get_hash(input_tensor, 'hash_input')
            if is_block_last:
                record['output_hash'] = self.get_hash(output_tensor, 'hash_output')
            copy_start = time.time()
            if is_global_first and should_save_activation:
                self.copy_tensor_to_record(input_tensor, record, 'input', 'input_async')
            if is_block_last and should_save_activation:
                self.copy_tensor_to_record(output_tensor, record, 'output', 'output_async')
            copy_time = time.time() - copy_start
            with self.stats_lock:
                self.time_stats['copy_forward_activations'] += copy_time
            
        return hook
    
    # At step-block start: hash/save weights and optimizer state
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
            self.copy_named_tensors_to_record(
                params_to_record, record, 'parameters', 'parameters_async',
                'copy_parameters',
            )
            if self.optimizer is not None:
                self.copy_optimizer_state_to_record(params_to_record, record)
        
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
    
    # Backward hook: hash/save grads at block boundaries, flush layer record
    def create_backward_hook(self, layer_name: str):
        if not self.enabled:
            def empty_hook(module, grad_input, grad_output):
                pass
            return empty_hook
        
        def hook(module, grad_input, grad_output):
            if self.layer_blocks_finalized and not self.is_block_first_layer(layer_name) and not self.is_block_last_layer(layer_name):
                return
            
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
                is_block_first = self.is_block_first_layer(layer_name)
                is_global_last = self.is_last_layer(layer_name)
                if is_block_first and input_grad is not None:
                    record['input_grad_hash'] = self.get_hash(input_grad, 'hash_input_grad')
                if is_global_last and output_grad is not None:
                    record['output_grad_hash'] = self.get_hash(output_grad, 'hash_output_grad')
                grad_cpu_copy_start = time.time()
                if is_block_first and input_grad is not None and should_save_grad:
                    self.copy_tensor_to_record(input_grad, record, 'input_grad', 'input_grad_async')
                if is_global_last and output_grad is not None and should_save_grad:
                    self.copy_tensor_to_record(output_grad, record, 'output_grad', 'output_grad_async')
                grad_cpu_copy_time = time.time() - grad_cpu_copy_start
                with self.stats_lock:
                    self.time_stats['copy_backward_grads'] += grad_cpu_copy_time
            
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
        record['input_ids_hash'] = self.get_hash(input_ids, 'hash_input')
        if labels is input_ids:
            record['labels_hash'] = record['input_ids_hash']
        else:
            record['labels_hash'] = self.get_hash(labels, 'hash_input')
        dataset_copy_start = time.time()
        self.copy_tensor_to_record(input_ids, record, 'input_ids', 'input_ids_async')
        self.copy_tensor_to_record(labels, record, 'labels', 'labels_async')
        dataset_copy_time = time.time() - dataset_copy_start
        with self.stats_lock:
            self.time_stats['copy_dataset_input'] += dataset_copy_time
        
    # Save module skeleton (no weights) for offline replay
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
        if layer_name == 'rotary_emb':
            torch.save(module, structure_path)
        else:
            buffer = io.BytesIO()
            torch.save(module, buffer)
            buffer.seek(0)
            meta_module = torch.load(buffer, map_location='meta', weights_only=False)
            torch.save(meta_module, structure_path)
        
        if original_device is not None and str(original_device) != 'cpu':
            module.to(original_device)
        
    
    def increment_step(self):
        if self.current_step in self.dataset_records:
            self.save_dataset_record(self.current_step)
            del self.dataset_records[self.current_step]
        
        self.current_step += 1
        self.tracked_layers.clear()
    
    # Map layers to layer_block_id and block_to_layers from layer_order
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
        return bool(self.layer_order) and layer_name == self.layer_order[0]
    
    def is_last_layer(self, layer_name: str):
        return bool(self.layer_order) and layer_name == self.layer_order[-1]
    
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
    
    def save_dataset_record_sync(self, step: int, record: dict, record_time: bool = True):
        block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        
        block_dir = os.path.join(self.save_dir, f"dataset_block_{block_idx}")
        os.makedirs(block_dir, exist_ok=True)
        
        step_dir = os.path.join(block_dir, f"step_{step_in_block}")
        os.makedirs(step_dir, exist_ok=True)
        
        if 'input_ids_async' in record:
            self.cuda_sync_for_save(record_time)
            input_ids = record['input_ids_async']
            labels = record['labels_async']
        else:
            input_ids = record['input_ids'] if 'input_ids' in record else None
            labels = record['labels'] if 'labels' in record else None
        
        input_ids_hash = record['input_ids_hash'] if 'input_ids_hash' in record else ''
        labels_hash = record['labels_hash'] if 'labels_hash' in record else ''
        if input_ids is not None:
            self.save_tensor_compressed(input_ids, os.path.join(step_dir, 'input_ids.pt'))
        if labels is not None:
            self.save_tensor_compressed(labels, os.path.join(step_dir, 'labels.pt'))
        output_logits = record['output_logits'] if 'output_logits' in record else None
        if output_logits is not None:
            self.save_tensor_compressed(output_logits, os.path.join(step_dir, 'output_logits.pt'))
        hash_info = {
            'input_ids_hash': input_ids_hash,
            'labels_hash': labels_hash,
            'loss': record['loss'] if 'loss' in record else None,
        }
        if 'output_logits_hash' in record:
            hash_info['output_logits_hash'] = record['output_logits_hash']
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
        
        def save_sync(record_copy):
            self.save_dataset_record_sync(step, record_copy, not self.async_save)
        self.enqueue_async_save(record, save_sync)
    
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
            self.cuda_sync_for_save(record_time)
            input_data = record['input_async'] if 'input_async' in record else None
            output_data = record['output_async'] if 'output_async' in record else None
        else:
            input_data = record['input'] if 'input' in record else None
            output_data = record['output'] if 'output' in record else None
        
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
            self.cuda_sync_for_save(record_time)
            input_grad_data = record['input_grad_async'] if 'input_grad_async' in record else None
            output_grad_data = record['output_grad_async'] if 'output_grad_async' in record else None
        else:
            input_grad_data = record['input_grad'] if 'input_grad' in record else None
            output_grad_data = record['output_grad'] if 'output_grad' in record else None
        
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
                self.cuda_sync_for_save(record_time)
                parameters = record['parameters_async']
            else:
                parameters = record['parameters'] if 'parameters' in record else None
            
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
                self.cuda_sync_for_save(record_time)
                optimizer_state = record['optimizer_state_async']
            else:
                optimizer_state = record['optimizer_state'] if 'optimizer_state' in record else None
            
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
    
    def save_layer_record(self, layer_name: str, step: int, save_hashes: bool = True):
        if self.warmup_mode:
            return
        
        if layer_name not in self.records or step not in self.records[layer_name]:
            return
        
        record = self.records[layer_name][step]
        
        def save_sync(record_copy):
            self.save_layer_record_sync(layer_name, step, record_copy, not self.async_save, save_hashes)
        self.enqueue_async_save(record, save_sync)
    
    def print_time_stats(self):
        print("\nTime statistics:")
        copy_keys = [
            'copy_forward_activations', 'copy_backward_grads', 'copy_parameters',
            'copy_optimizer_state', 'copy_dataset_input',
        ]
        hash_categories = [
            ('hash_input', 'Input Hash', 'count_hash_input'),
            ('hash_output', 'Output Hash', 'count_hash_output'),
            ('hash_input_grad', 'Input Grad Hash', 'count_hash_input_grad'),
            ('hash_output_grad', 'Output Grad Hash', 'count_hash_output_grad'),
            ('hash_parameters', 'Parameter Hash', 'count_hash_parameters'),
            ('hash_optimizer_state', 'Optimizer State Hash', 'count_hash_optimizer_state'),
        ]
        save_keys = [
            'save_activations', 'save_grads', 'save_parameters',
            'save_optimizer_state', 'save_hashes',
        ]
        total_copy_time = sum(self.time_stats[k] for k in copy_keys)
        total_hash_time = sum(self.time_stats[k] for k, _, _ in hash_categories)
        total_hash_count = sum(self.time_stats[k] for _, _, k in hash_categories)
        total_disk_save = sum(self.time_stats[k] for k in save_keys)
        sync_wait_time = self.time_stats['async_submit'] + self.time_stats['backpressure_wait']
        if self.async_save:
            total_blocking_time = total_copy_time + total_hash_time + sync_wait_time
        else:
            total_blocking_time = total_copy_time + total_hash_time + total_disk_save + self.time_stats['cuda_synchronize']

        print(f"  I/O: {total_copy_time:.3f} seconds")
        print("  Hash:")
        for time_key, label, count_key in hash_categories:
            count_val = self.time_stats[count_key]
            if count_val > 0:
                time_val = self.time_stats[time_key]
                avg_ms = time_val / count_val * 1000
                print(f"    {label}: {time_val:.3f} seconds ({count_val} times, avg {avg_ms:.2f} ms)")
        print(f"    Total: {total_hash_time:.3f} seconds ({total_hash_count} times)")
        if not self.async_save:
            print(f"  Disk save: {total_disk_save:.3f} seconds")
        print(f"  Countable overhead: {total_blocking_time:.3f} seconds")
    
    def wait_all_tasks(self):
        if self.executor is not None and self.pending_futures:
            for future in self.pending_futures:
                future.result()
            self.pending_futures.clear()
        for process in self.compress_processes:
            process.wait()
        self.compress_processes.clear()
    
    # Export trained weights to final_model/ after training
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
            'hashes': {'size': 0, 'count': 0},
            'other': {'size': 0, 'count': 0},
        }
        
        final_model_dir = os.path.join(self.save_dir, 'final_model')
        for root, dirs, files in os.walk(self.save_dir):
            if root.startswith(final_model_dir):
                continue
            for file in files:
                filepath = os.path.join(root, file)
                file_size = os.path.getsize(filepath)
                category = self.storage_file_category(file, root)
                storage_stats[category]['size'] += file_size
                storage_stats[category]['count'] += 1
        
        total_size = sum(info['size'] for info in storage_stats.values())
        
        print(f"  Total usage: {total_size / (1024**3):.2f} GB")
        
        categories_display = [
            ('activations', 'Activations'),
            ('grads', 'Gradients'),
            ('parameters', 'Parameters'),
            ('optimizer_state', 'Optimizer State'),
            ('hashes', 'Hash records'),
            ('other', 'Other'),
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
    
    # Flush pending records and write metadata.json
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
        for key, filename in (
            ('input_ids', 'input_ids.pt'),
            ('labels', 'labels.pt'),
            ('output_logits', 'output_logits.pt'),
        ):
            path = os.path.join(step_dir, filename)
            if self.file_exists(path):
                record[key] = self.load_tensor_compressed(path).to(device)
        
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
            self.layer_to_block = metadata['layer_to_block']
            self.block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
            self.layer_blocks_finalized = True
    
    def file_exists(self, path):
        return os.path.exists(path) or os.path.exists(path + '.zst')
    
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
        
        if layer_name == 'rotary_emb':
            return torch.load(structure_path, map_location=device, weights_only=False)
        module = torch.load(structure_path, map_location='meta', weights_only=False)
        module.to_empty(device=device)
        return module
    
    # Load one step's layer record (params, hashes, optimizer, activations)
    def load_layer_record(self, layer_name: str, step: int, device):
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        
        layer_block_idx = self.layer_to_block.get(layer_name, 0) if self.layer_blocks_finalized else 0
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        if not os.path.exists(block_dir):
            raise FileNotFoundError(f"Block not found: {block_dir}")

        block_layers = self.block_to_layers.get(layer_block_idx, [])
        is_internal_layer = (
            layer_name in block_layers
            and layer_name != block_layers[0]
            and layer_name != block_layers[-1]
        )

        layer_dir = os.path.join(block_dir, layer_name)
        if not os.path.exists(layer_dir):
            if is_internal_layer:
                return {}
            raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        if not os.path.exists(step_dir):
            if is_internal_layer:
                return {}
            raise FileNotFoundError(f"Step not found: {step_dir}")

        record = {}
        for key, filename in (
            ('input', 'input.pt'),
            ('output', 'output.pt'),
            ('input_grad', 'input_grad.pt'),
            ('output_grad', 'output_grad.pt'),
        ):
            path = os.path.join(step_dir, filename)
            if self.file_exists(path):
                record[key] = self.load_tensor_compressed(path).to(device)

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

    def load_layer_hashes(self, layer_name, step):
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block

        layer_block_idx = self.layer_to_block.get(layer_name, 0) if self.layer_blocks_finalized else 0

        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        if not os.path.exists(block_dir):
            raise FileNotFoundError(f"Block not found: {block_dir}")

        block_layers = self.block_to_layers.get(layer_block_idx, [])
        is_internal_layer = (
            layer_name in block_layers
            and layer_name != block_layers[0]
            and layer_name != block_layers[-1]
        )

        layer_dir = os.path.join(block_dir, layer_name)
        if not os.path.exists(layer_dir):
            if is_internal_layer:
                return {}
            raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        if not os.path.exists(step_dir):
            if is_internal_layer:
                return {}
            raise FileNotFoundError(f"Step not found: {step_dir}")

        hash_file = os.path.join(step_dir, 'hashes.json')
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                return json.load(f)
        return {}
    
    def load_layer_input(self, layer_name: str, step: int, device):
        self.ensure_metadata_loaded()
        
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        layer_block_idx = self.layer_to_block.get(layer_name, 0)
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        layer_dir = os.path.join(block_dir, layer_name)
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        input_path = os.path.join(step_dir, 'input.pt')
        
        if self.is_first_layer(layer_name):
            if self.file_exists(input_path):
                tensor = self.load_tensor_compressed(input_path)
                return tensor.to(device)
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
        return tensor.to(device)
    
    def load_layer_output_grad(self, layer_name: str, step: int, device):
        self.ensure_metadata_loaded()
        
        step_block_idx = step // self.steps_per_block
        step_in_block = step % self.steps_per_block
        layer_block_idx = self.layer_to_block.get(layer_name, 0)
        
        block_dir = os.path.join(self.save_dir, f"block_{layer_block_idx}_{step_block_idx}")
        layer_dir = os.path.join(block_dir, layer_name)
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        output_grad_path = os.path.join(step_dir, 'output_grad.pt')
        
        if self.is_last_layer(layer_name):
            if self.file_exists(output_grad_path):
                tensor = self.load_tensor_compressed(output_grad_path)
                return tensor.to(device)
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
        return tensor.to(device)
