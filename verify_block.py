import torch
import torch.optim as optim
import argparse
import os
import json
import time
from tqdm import tqdm
from layer_recorder import LayerRecorder, set_deterministic
import aftune_torch

def get_hash(tensor, chunk_size, use_blake3):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    root = aftune_torch.ops.flat_root(tensor, chunk_size=chunk_size, use_blake3=use_blake3)
    return root.hex()

def verify_block(layer_block_id, step_block_id, recorder, device, strict_hash):
    metadata = recorder.load_metadata()
    
    steps_per_block = metadata['steps_per_block']
    checkpoint_interval = metadata['checkpoint_interval']
    chunk_size = metadata['chunk_size']
    use_blake3 = metadata['use_blake3']
    total_steps = metadata['total_steps']
    layer_order = metadata['layer_order']
    optimizer_type = metadata['optimizer_type']
    block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
    model_name = metadata['model_name']
    learning_rate = metadata['learning_rate']
    
    if layer_block_id not in block_to_layers:
        return {'error': f'Layer Block {layer_block_id} does not exist'}
    
    layer_names = block_to_layers[layer_block_id]
    
    is_last_block = ('lm_head' in layer_names)
    
    
    layers = {}
    all_params = []
    for layer_name in layer_names:
        layer_module = recorder.load_module_structure(layer_name, device=device)
        layers[layer_name] = layer_module
        all_params.extend(layer_module.parameters())
    
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(all_params, lr=learning_rate)
    else:
        optimizer = optim.AdamW(all_params, lr=learning_rate)
    
    time_stats = {
        'load_data': 0.0,
        'hash_verification': 0.0,
        'loss_verification': 0.0,
        'forward_pass': 0.0,
        'backward_pass': 0.0,
        'optimizer_step': 0.0,
        'weight_comparison': 0.0
    }
    
    block_start_step = step_block_id * steps_per_block
    block_end_step = (step_block_id + 1) * steps_per_block
    
    if total_steps is not None:
        block_end_step = min(block_end_step, total_steps)
        if block_start_step >= total_steps:
            return {'error': f'Step Block {step_block_id} exceeds training range (total_steps={total_steps})'}
    
    
    load_start = time.time()
    
    use_lora = metadata['use_lora']
    first_layer_record = recorder.load_layer_record(layer_names[0], block_start_step, device=device)
    
    if 'parameters' not in first_layer_record:
        if checkpoint_interval > 1:
            source_step_block = (step_block_id // checkpoint_interval) * checkpoint_interval
            error_msg = f'Step Block {step_block_id} does not have complete checkpoint (checkpoint_interval={checkpoint_interval})\n'
            error_msg += f'\nPlease generate checkpoint on GPU first:\n'
            error_msg += f'  python prepare_block.py --layer_block_id {layer_block_id} --step_block_id {step_block_id}\n'
            error_msg += f'\nThis command will replay from Step Block {source_step_block} to Step Block {step_block_id} and save checkpoint'
            del first_layer_record
            return {'error': error_msg}
        else:
            del first_layer_record
            return {'error': f'{layer_names[0]} has no parameter record at step {block_start_step}'}
    
    del first_layer_record
    
    for layer_name in layer_names:
        layer_record = recorder.load_layer_record(layer_name, block_start_step, device=device)
        
        layer_has_params = len(list(layers[layer_name].parameters())) > 0
        
        if 'parameters' not in layer_record:
            if layer_has_params:
                if not use_lora:
                    return {'error': f'{layer_name} has no parameter record at step {block_start_step}'}
            continue
        else:
            hash_start = time.time()
            for param_name, param_tensor in layer_record['parameters'].items():
                expected_hash = layer_record['parameter_hashes'][param_name]
                actual_hash = get_hash(param_tensor, chunk_size, use_blake3)
                if actual_hash != expected_hash:
                    error_msg = f'{layer_name}.{param_name} hash mismatch (Step {block_start_step})\n'
                    error_msg += f'  Expected hash: {expected_hash}\n'
                    error_msg += f'  Actual hash: {actual_hash}\n'
                    return {'error': error_msg}
                
                for name, param in layers[layer_name].named_parameters():
                    if name == param_name:
                        param.data.copy_(param_tensor)
                        break
            
            del layer_record['parameters']
            time_stats['hash_verification'] += time.time() - hash_start
        
        if 'optimizer_state' in layer_record:
            hash_start = time.time()
            for param_name, state_dict in layer_record['optimizer_state'].items():
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor):
                        hash_key = f"{param_name}.{k}"
                        expected_hash = layer_record['optimizer_state_hashes'][hash_key]
                        actual_hash = get_hash(v, chunk_size, use_blake3)
                        if actual_hash != expected_hash:
                            error_msg = f'{layer_name}.{hash_key} optimizer state hash mismatch (Step {block_start_step})\n'
                            error_msg += f'  Expected hash: {expected_hash}\n'
                            error_msg += f'  Actual hash: {actual_hash}\n'
                            return {'error': error_msg}
                
                for name, param in layers[layer_name].named_parameters():
                    if name == param_name:
                        restored_state = {}
                        for k, v in state_dict.items():
                            if isinstance(v, torch.Tensor):
                                if k == 'step':
                                    restored_state[k] = v
                                else:
                                    restored_state[k] = v.to(dtype=param.dtype, device=device)
                            else:
                                restored_state[k] = v
                        optimizer.state[param] = restored_state
                        break
            del layer_record['optimizer_state']
            time_stats['hash_verification'] += time.time() - hash_start
        
        del layer_record
    time_stats['load_data'] += time.time() - load_start
    
    step_results = []
    
    pbar = tqdm(range(block_start_step, block_end_step), desc=f"Replay Layer Block {layer_block_id} × Step Block {step_block_id}", unit="step")
    
    for step in pbar:
        load_start = time.time()
        step_records = {}
        for layer_name in layer_names:
            step_records[layer_name] = recorder.load_layer_record(layer_name, step, device=device)
        time_stats['load_data'] += time.time() - load_start
        
        ds_record = None
        if is_last_block:
            ds_record = recorder.load_dataset_record(step, device=device)
        
        loss_verification_result = None
        if is_last_block:
            loss_start = time.time()
            expected_loss = ds_record.get('loss')
            
            labels = ds_record['labels'].to(device)
            logits = step_records['lm_head']['output'].to(device).to(torch.float32)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            computed_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            if expected_loss is not None:
                loss_diff = abs(computed_loss.item() - expected_loss)
                loss_verification_result = {
                    'expected': expected_loss,
                    'computed': computed_loss.item(),
                    'diff': loss_diff,
                    'passed': loss_diff <= 1e-3
                }
                if loss_diff > 1e-3:
                    pbar.close()
                    return {'error': f'Loss verification failed at step {step}: difference {loss_diff:.6e}'}
            
            del logits, shift_logits, shift_labels, computed_loss
            time_stats['loss_verification'] += time.time() - loss_start
        
        layer_outputs = {}
        layer_inputs = {}
        forward_metrics_per_layer = {}
        backward_metrics_per_layer = {}
        
        is_block_first_layer = {layer_name: (layer_names.index(layer_name) == 0) for layer_name in layer_names}
        is_block_last_layer = {layer_name: (layer_names.index(layer_name) == len(layer_names) - 1) for layer_name in layer_names}
        is_global_first_layer = {layer_name: (layer_order.index(layer_name) == 0) for layer_name in layer_names}
        
        for idx, layer_name in enumerate(layer_names):
            step_record = step_records[layer_name]
            layer_module = layers[layer_name]
            
            if is_block_first_layer[layer_name]:
                hash_start = time.time()
                input_tensor = recorder.load_layer_input(layer_name, step, device=device)
                
                if is_global_first_layer[layer_name] and 'input_hash' in step_record:
                    expected_input_hash = step_record['input_hash']
                    actual_input_hash = get_hash(input_tensor, chunk_size, use_blake3)
                    if actual_input_hash != expected_input_hash:
                        pbar.close()
                        return {'error': f'{layer_name}: Input hash mismatch (step {step})'}
                time_stats['hash_verification'] += time.time() - hash_start
            else:
                prev_layer_name = layer_names[idx - 1]
                input_tensor = layer_outputs[prev_layer_name]
                if isinstance(input_tensor, tuple):
                    input_tensor = input_tensor[0]
            
            forward_start = time.time()
            layer_input = input_tensor
            
            if layer_name == "embedding":
                if layer_input.dtype == torch.long:
                    computed_output = layer_module(layer_input)
                else:
                    computed_output = layer_input
                layer_inputs[layer_name] = layer_input
            elif layer_name.startswith("layer_"):
                input_grad_tensor = layer_input.clone().requires_grad_(True)
                input_grad_tensor.retain_grad()
                position_ids = torch.arange(0, input_grad_tensor.shape[1], device=device).unsqueeze(0)
                output = layer_module(input_grad_tensor, position_ids=position_ids)
                computed_output = output[0] if isinstance(output, tuple) else output
                layer_inputs[layer_name] = input_grad_tensor
            elif layer_name in ["norm", "lm_head"]:
                input_grad_tensor = layer_input.clone().requires_grad_(True)
                input_grad_tensor.retain_grad()
                computed_output = layer_module(input_grad_tensor)
                layer_inputs[layer_name] = input_grad_tensor
            elif layer_name == "classifier":
                if model_name == "dinov2_giant" and len(layer_input.shape) == 3:
                    cls_token = layer_input[:, 0]
                    patch_tokens = layer_input[:, 1:]
                    patch_mean = patch_tokens.mean(dim=1)
                    processed_input = torch.cat([cls_token, patch_mean], dim=1)
                elif model_name == "vit_large" and len(layer_input.shape) == 3:
                    processed_input = layer_input[:, 0, :]
                else:
                    processed_input = layer_input
                
                input_grad_tensor = processed_input.clone().requires_grad_(True)
                input_grad_tensor.retain_grad()
                computed_output = layer_module(input_grad_tensor)
                layer_inputs[layer_name] = input_grad_tensor
            else:
                input_grad_tensor = layer_input.clone().requires_grad_(True)
                input_grad_tensor.retain_grad()
                computed_output = layer_module(input_grad_tensor)
                layer_inputs[layer_name] = input_grad_tensor
            
            time_stats['forward_pass'] += time.time() - forward_start
            
            if isinstance(computed_output, tuple):
                computed_output = computed_output[0]
            layer_outputs[layer_name] = computed_output
            
            if is_block_last_layer[layer_name]:
                hash_start = time.time()
                expected_output = recorder.load_layer_output(layer_name, step, device=device)
                
                if 'output_hash' in step_record:
                    expected_output_hash = step_record['output_hash']
                    actual_output_hash = get_hash(expected_output, chunk_size, use_blake3)
                    if actual_output_hash != expected_output_hash:
                        pbar.close()
                        return {'error': f'{layer_name}: Output hash mismatch (disk data corrupted, step {step})'}
                
                time_stats['hash_verification'] += time.time() - hash_start
                
                output_diff = torch.abs(computed_output - expected_output)
                forward_metrics_per_layer[layer_name] = {
                    'max_abs': output_diff.max().item(),
                    'mean_abs': output_diff.mean().item(),
                    'relative_l2': (torch.norm(output_diff) / torch.norm(expected_output)).item()
                }
                del expected_output, output_diff
        
        if forward_metrics_per_layer:
            avg_fwd_l2 = sum(m['relative_l2'] for m in forward_metrics_per_layer.values()) / len(forward_metrics_per_layer)
            pbar.set_postfix({'avg_fwd_l2': f"{avg_fwd_l2:.2e}"})
        
        first_layer = layer_names[0]
        last_layer = layer_names[-1]
        
        backward_start = time.time()
        optimizer.zero_grad()
        
        if not is_last_block:
            hash_start = time.time()
            expected_output_grad = recorder.load_layer_output_grad(last_layer, step, device=device)
            time_stats['hash_verification'] += time.time() - hash_start
            layer_outputs[last_layer].backward(expected_output_grad, retain_graph=True)
            del expected_output_grad
        else:
            if last_layer == 'lm_head':
                labels = ds_record['labels'].to(device)
                logits = layer_outputs['lm_head'].to(torch.float32)
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                computed_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                computed_loss.backward()
                del logits, shift_logits, shift_labels, computed_loss
            else:
                hash_start = time.time()
                expected_output_grad = recorder.load_layer_output_grad(last_layer, step, device=device)
                time_stats['hash_verification'] += time.time() - hash_start
                layer_outputs[last_layer].backward(expected_output_grad, retain_graph=True)
                del expected_output_grad
        
        time_stats['backward_pass'] += time.time() - backward_start
        
        if 'input_grad_hash' in step_records[first_layer]:
            hash_start = time.time()
            expected_input_grad = recorder.load_layer_input_grad(first_layer, step, device=device)
            
            expected_hash = step_records[first_layer]['input_grad_hash']
            actual_hash = get_hash(expected_input_grad, chunk_size, use_blake3)
            if actual_hash != expected_hash:
                pbar.close()
                return {'error': f'{first_layer}: Input grad hash mismatch (step {step})'}
            
            if first_layer in layer_inputs and hasattr(layer_inputs[first_layer], 'grad') and layer_inputs[first_layer].grad is not None:
                computed_grad = layer_inputs[first_layer].grad
                grad_diff = torch.abs(computed_grad - expected_input_grad)
                rel_l2 = (torch.norm(grad_diff) / (torch.norm(expected_input_grad) + 1e-10)).item()
                
                backward_metrics_per_layer[first_layer] = {
                    'max_abs': grad_diff.max().item(),
                    'mean_abs': grad_diff.mean().item(),
                    'relative_l2': rel_l2
                }
                del expected_input_grad, computed_grad, grad_diff
            
            time_stats['hash_verification'] += time.time() - hash_start
        
        opt_start = time.time()
        optimizer.step()
        time_stats['optimizer_step'] += time.time() - opt_start
        
        step_results.append({
            'step': step,
            'forward_per_layer': forward_metrics_per_layer,
            'backward_per_layer': backward_metrics_per_layer,
            'loss_verification': loss_verification_result
        })
        
        del step_records, layer_outputs, layer_inputs
        if ds_record is not None:
            del ds_record
    
    pbar.close()
    
    weight_verification = None
    optimizer_state_verification = None

    compare_source = None
    is_last_step_block = (total_steps is not None and block_end_step >= total_steps)
    if not is_last_step_block:
        compare_source = 'next_block'
    else:
        if os.path.exists(os.path.join(recorder.save_dir, 'final_model')):
            compare_source = 'final_model'

    records_for_compare = None
    if compare_source == 'next_block':
        load_start = time.time()
        next_block_start = (step_block_id + 1) * steps_per_block
        records_for_compare = {}
        for layer_name in layer_names:
            layer_record = recorder.load_layer_record(layer_name, next_block_start, device=device)
            records_for_compare[layer_name] = {
                'parameters': layer_record.get('parameters', {}),
                'parameter_hashes': layer_record.get('parameter_hashes', {}),
                'optimizer_state': layer_record.get('optimizer_state', {}),
                'optimizer_state_hashes': layer_record.get('optimizer_state_hashes', {})
            }
            del layer_record
        time_stats['load_data'] += time.time() - load_start
    elif compare_source == 'final_model':
        load_start = time.time()
        records_for_compare = {}
        final_dir = os.path.join(recorder.save_dir, 'final_model')
        for layer_name in layer_names:
            layer_dir = os.path.join(final_dir, layer_name)
            if not os.path.exists(layer_dir):
                continue
            params_dir = os.path.join(layer_dir, 'parameters')
            if os.path.exists(params_dir):
                records_for_compare[layer_name] = {'parameters': {}}
                with open(os.path.join(params_dir, 'name_mapping.json'), 'r') as f:
                    name_mapping = json.load(f)
                for safe_name, original_name in name_mapping.items():
                    p = os.path.join(params_dir, f'{safe_name}.pt')
                    if os.path.exists(p):
                        records_for_compare[layer_name]['parameters'][original_name] = torch.load(p, map_location=device, weights_only=False)
            hash_file = os.path.join(layer_dir, 'parameter_hashes.json')
            if os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    records_for_compare[layer_name]['parameter_hashes'] = json.load(f)
        time_stats['load_data'] += time.time() - load_start

    if compare_source is not None and records_for_compare is not None:
        def verify_tensor_diff(actual, expected, key, expected_hash):
            if strict_hash:
                replayed_hash = get_hash(actual, chunk_size, use_blake3)
                hash_match = (replayed_hash == expected_hash)
                result = {
                    'hash_match': hash_match,
                    'expected_hash': expected_hash,
                    'replayed_hash': replayed_hash
                }
                if not hash_match:
                    return result, f'{key} hash mismatch after replay (not bitwise identical)'
                return result, None
            else:
                chunk_size = 1024 * 1024
                total_elements = actual.numel()
                
                max_abs = 0.0
                sum_abs = 0.0
                sum_diff_sq = 0.0
                sum_expected_sq = 0.0
                
                actual_flat = actual.flatten()
                expected_flat = expected.flatten()
                
                for i in range(0, total_elements, chunk_size):
                    end_idx = min(i + chunk_size, total_elements)
                    actual_chunk = actual_flat[i:end_idx]
                    expected_chunk = expected_flat[i:end_idx]
                    
                    diff_chunk = (actual_chunk - expected_chunk).abs()
                    
                    max_abs = max(max_abs, diff_chunk.max().item())
                    sum_abs += diff_chunk.sum().item()
                    sum_diff_sq += torch.sum(diff_chunk * diff_chunk).item()
                    sum_expected_sq += torch.sum(expected_chunk * expected_chunk).item()
                    
                    del diff_chunk, actual_chunk, expected_chunk
                
                mean_abs = sum_abs / total_elements
                diff_norm = (sum_diff_sq ** 0.5)
                expected_norm = (sum_expected_sq ** 0.5)
                rel_l2 = diff_norm / (expected_norm + 1e-10)
                
                return {
                    'max_abs': max_abs,
                    'mean_abs': mean_abs,
                    'relative_l2': rel_l2,
                    'diff_norm_sq': sum_diff_sq,
                    'expected_norm_sq': sum_expected_sq
                }, None

        comp_start = time.time()
        weight_diffs = {}
        layer_overall_l2 = {}
        hash_start = time.time()
        for layer_name in layer_names:
            rec = records_for_compare.get(layer_name)
            if not rec or 'parameters' not in rec:
                continue
            layer_module = layers[layer_name]
            layer_diff_norm_sq = 0.0
            layer_expected_norm_sq = 0.0
            param_names = list(rec['parameters'].keys())
            for param_name in param_names:
                expected_param = rec['parameters'][param_name]
                expected_hash = rec['parameter_hashes'][param_name]
                actual_hash = get_hash(expected_param, chunk_size, use_blake3)
                if actual_hash != expected_hash:
                    src = 'final model' if compare_source == 'final_model' else 'next block start'
                    return {'error': f'{layer_name}.{param_name} hash mismatch ({src})'}
                
                del actual_hash
                
                for name, layer_param in layer_module.named_parameters():
                    if name == param_name:
                        param_key = f"{layer_name}.{param_name}"
                        diff_result, error = verify_tensor_diff(layer_param, expected_param, param_key, expected_hash)
                        weight_diffs[param_key] = diff_result
                        if error:
                            return {'error': error}
                        
                        if 'diff_norm_sq' in diff_result and 'expected_norm_sq' in diff_result:
                            layer_diff_norm_sq += diff_result['diff_norm_sq']
                            layer_expected_norm_sq += diff_result['expected_norm_sq']
                        
                        del expected_param
                        break
                
                del rec['parameters'][param_name]
            
            if layer_expected_norm_sq > 0:
                layer_overall_l2[layer_name] = (layer_diff_norm_sq ** 0.5) / (layer_expected_norm_sq ** 0.5 + 1e-10)
            else:
                layer_overall_l2[layer_name] = 0.0
        
        time_stats['hash_verification'] += time.time() - hash_start
        
        all_diff_norm_sq = 0.0
        all_expected_norm_sq = 0.0
        for diff_result in weight_diffs.values():
            if 'diff_norm_sq' in diff_result and 'expected_norm_sq' in diff_result:
                all_diff_norm_sq += diff_result['diff_norm_sq']
                all_expected_norm_sq += diff_result['expected_norm_sq']
        
        if all_expected_norm_sq > 0:
            overall_relative_l2 = (all_diff_norm_sq ** 0.5) / (all_expected_norm_sq ** 0.5 + 1e-10)
        else:
            overall_relative_l2 = 0.0
        optimizer_state_diffs = {}
        if compare_source == 'next_block':
            for layer_name in layer_names:
                rec = records_for_compare.get(layer_name)
                if not rec or 'optimizer_state' not in rec:
                    continue
                layer_module = layers[layer_name]
                param_names = list(rec['optimizer_state'].keys())
                for param_name in param_names:
                    state_dict = rec['optimizer_state'][param_name]
                    target_param = None
                    for name, param in layer_module.named_parameters():
                        if name == param_name:
                            target_param = param
                            break
                    if target_param is None or target_param not in optimizer.state:
                        continue
                    actual_state = optimizer.state[target_param]
                    for k, expected_v in state_dict.items():
                        if isinstance(expected_v, torch.Tensor) and k in actual_state:
                            actual_v = actual_state[k]
                            state_key = f"{layer_name}.{param_name}.{k}"
                            expected_hash = rec['optimizer_state_hashes'][f"{param_name}.{k}"]
                            diff_result, error = verify_tensor_diff(actual_v, expected_v, state_key, expected_hash)
                            optimizer_state_diffs[state_key] = diff_result
                            if error:
                                return {'error': error}
                            del expected_v
                    del rec['optimizer_state'][param_name]
        time_stats['weight_comparison'] += time.time() - comp_start
        
        del records_for_compare

        if compare_source == 'next_block':
            weight_verification = {
                'next_step_block_id': step_block_id + 1,
                'param_diffs': weight_diffs,
                'overall_relative_l2': overall_relative_l2,
                'layer_overall_l2': layer_overall_l2
            }
            if optimizer_state_diffs:
                if strict_hash:
                    all_match = [v['hash_match'] for v in optimizer_state_diffs.values()]
                    optimizer_state_verification = {
                        'mode': 'strict_hash',
                        'total': len(all_match),
                        'matched': sum(all_match),
                        'all_match': all(all_match),
                        'details': optimizer_state_diffs
                    }
                else:
                    all_max = [v['max_abs'] for v in optimizer_state_diffs.values()]
                    all_mean = [v['mean_abs'] for v in optimizer_state_diffs.values()]
                    all_rel = [v['relative_l2'] for v in optimizer_state_diffs.values()]
                    optimizer_state_verification = {
                        'mode': 'numeric',
                        'max_abs': max(all_max) if all_max else 0.0,
                        'mean_abs': sum(all_mean) / len(all_mean) if all_mean else 0.0,
                        'relative_l2': sum(all_rel) / len(all_rel) if all_rel else 0.0,
                        'details': optimizer_state_diffs
                    }
        else:
            weight_verification = {
                'source': 'final_model',
                'param_diffs': weight_diffs,
                'overall_relative_l2': overall_relative_l2,
                'layer_overall_l2': layer_overall_l2
            }
    
    del layers
    del optimizer
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    return {
        'layer_block_id': layer_block_id,
        'step_block_id': step_block_id,
        'layer_names': layer_names,
        'steps': step_results,
        'weight_verification': weight_verification,
        'optimizer_state_verification': optimizer_state_verification,
        'is_last_block': is_last_block,
        'time_stats': time_stats,
        'actual_steps': len(step_results),
        'expected_steps': block_end_step - block_start_step
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', type=str, default='./block_records')
    parser.add_argument('-l', '--layer_block_id', type=int, required=True,
                        help='Layer Block ID')
    parser.add_argument('-s', '--step_block_id', type=int, required=True,
                        help='Step Block ID')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--strict_hash', action='store_true',
                        help='Use strict hash verification')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_deterministic(seed=args.seed, enabled=args.deterministic)
    
    temp_recorder = LayerRecorder(save_dir=args.record_dir)
    metadata = temp_recorder.load_metadata()
    
    recorder = LayerRecorder(
        save_dir=args.record_dir,
        steps_per_block=metadata['steps_per_block'],
        layers_per_block=metadata['layers_per_block'],
        checkpoint_interval=metadata['checkpoint_interval'],
        activation_interval=metadata['activation_interval'],
        enabled=metadata['enabled'],
        async_save=metadata['async_save'],
        chunk_size=metadata['chunk_size'],
        use_blake3=metadata['use_blake3'],
        learning_rate=metadata['learning_rate'],
        optimizer_type=metadata['optimizer_type'],
        module_structure_dir=metadata['module_structure_dir'],
        model_name=metadata['model_name']
    )
    
    recorder.layer_to_block = metadata['layer_to_block']
    recorder.block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
    recorder.layer_order = metadata['layer_order']
    recorder.layer_order_finalized = True
    recorder.layer_blocks_finalized = True
    recorder.use_lora = metadata['use_lora']
    
    result = verify_block(args.layer_block_id, args.step_block_id, recorder, args.device, args.strict_hash)
    
    if 'error' in result:
        raise RuntimeError(result['error'])
    
    time_stats = result['time_stats']
    total_time = sum(time_stats.values())
    if total_time > 0:
        print(f"\nVerification time: {total_time:.3f} seconds")
    
    

if __name__ == '__main__':
    main()
