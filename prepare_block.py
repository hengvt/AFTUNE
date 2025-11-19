import torch
import torch.optim as optim
import argparse
import os
import json
import time
from tqdm import tqdm
from layer_recorder import LayerRecorder, set_deterministic
from verify_block import get_hash
import aftune_torch


def save_step_activations_and_grads(recorder, layer_names, layer_block_id, step_block_id,
                                     step_in_block, layer_inputs, layer_outputs, layer_order):
    block_dir = os.path.join(recorder.save_dir, f"block_{layer_block_id}_{step_block_id}")
    os.makedirs(block_dir, exist_ok=True)
    
    is_global_first = (layer_names[0] == layer_order[0])
    verification_results = {}
    
    for layer_name in layer_names:
        layer_dir = os.path.join(block_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        
        step_dir = os.path.join(layer_dir, f"step_{step_in_block}")
        os.makedirs(step_dir, exist_ok=True)
        
        verification_results[layer_name] = {}
        
        hash_file = os.path.join(step_dir, 'hashes.json')
        expected_hashes = {}
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                expected_hashes = json.load(f)
        
        def save_and_verify(tensor, filename, hash_key):
            if tensor is not None:
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                filepath = os.path.join(step_dir, filename)
                recorder.save_tensor_compressed(tensor.detach().cpu(), filepath)
                actual_hash = get_hash(tensor, recorder.chunk_size, recorder.use_blake3)
                expected_hash = expected_hashes.get(hash_key, '')
                verification_results[layer_name][hash_key.replace('_hash', '')] = (
                    actual_hash == expected_hash, expected_hash, actual_hash
                )
        
        is_block_first = (layer_name == layer_names[0])
        is_block_last = (layer_name == layer_names[-1])
        is_global_last = (layer_name == layer_order[-1])
        
        if not recorder.deduplicate_storage or (is_global_first and is_block_first):
            save_and_verify(layer_inputs.get(layer_name), 'input.pt', 'input_hash')
        if not recorder.deduplicate_storage or is_block_last:
            save_and_verify(layer_outputs.get(layer_name), 'output.pt', 'output_hash')
        
        input_tensor = layer_inputs.get(layer_name)
        output_tensor = layer_outputs.get(layer_name)
        if not recorder.deduplicate_storage or is_block_first:
            if input_tensor is not None and hasattr(input_tensor, 'grad') and input_tensor.grad is not None:
                save_and_verify(input_tensor.grad, 'input_grad.pt', 'input_grad_hash')
        if not recorder.deduplicate_storage or is_global_last:
            if output_tensor is not None and hasattr(output_tensor, 'grad') and output_tensor.grad is not None:
                save_and_verify(output_tensor.grad, 'output_grad.pt', 'output_grad_hash')
    
    return verification_results

def prepare_checkpoint_for_block(layer_block_id, step_block_id, recorder, device, block_records_dir):
    metadata = recorder.load_metadata()
    
    steps_per_block = metadata['steps_per_block']
    layers_per_block = metadata['layers_per_block']
    checkpoint_interval = metadata['checkpoint_interval']
    activation_interval = metadata['activation_interval']
    layer_order = metadata['layer_order']
    optimizer_type = metadata['optimizer_type']
    learning_rate = metadata['learning_rate']
    block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
    model_name = metadata['model_name']
    
    block_checkpoint_interval = {int(k): v for k, v in metadata['block_checkpoint_interval'].items()}
    actual_checkpoint_interval = block_checkpoint_interval.get(layer_block_id, checkpoint_interval)
    
    return generate_single_layer_block(
        layer_block_id, step_block_id, recorder, device,
        steps_per_block, checkpoint_interval, activation_interval,
        layer_order, optimizer_type, learning_rate, block_to_layers, model_name, block_checkpoint_interval, block_records_dir
    )

def generate_single_layer_block(layer_block_id, step_block_id, recorder, device,
                                   steps_per_block, checkpoint_interval, activation_interval,
                                   layer_order, optimizer_type, learning_rate, block_to_layers, model_name, block_checkpoint_interval, block_records_dir):
    if layer_block_id not in block_to_layers:
        return {'error': f'Layer Block {layer_block_id} does not exist'}
    
    layer_names = block_to_layers[layer_block_id]
    target_step = step_block_id * steps_per_block
    
    first_layer = layer_names[0]
    last_layer = layer_names[-1]
    layer_block_idx = layer_block_id
    block_dir = os.path.join(recorder.save_dir, f"block_{layer_block_idx}_{step_block_id}")
    
    has_checkpoint = False
    has_activation = False
    
    if os.path.exists(block_dir):
        first_layer_dir = os.path.join(block_dir, first_layer)
        checkpoint_hash_file = os.path.join(first_layer_dir, 'checkpoint_hashes.json')
        
        if os.path.exists(checkpoint_hash_file):
            with open(checkpoint_hash_file, 'r') as f:
                checkpoint_info = json.load(f)
                has_checkpoint = checkpoint_info.get('has_checkpoint', False)
        
        last_layer_dir = os.path.join(block_dir, last_layer)
        step_dir = os.path.join(last_layer_dir, 'step_0')
        if os.path.exists(step_dir):
            output_path = os.path.join(step_dir, 'output.pt')
            input_path = os.path.join(step_dir, 'input.pt')
            has_activation = (os.path.exists(output_path) or os.path.exists(output_path + '.zst') or
                           os.path.exists(input_path) or os.path.exists(input_path + '.zst'))
    
    if block_checkpoint_interval is None:
        metadata = recorder.load_metadata()
        block_checkpoint_interval_dict = {int(k): v for k, v in metadata['block_checkpoint_interval'].items()}
    else:
        block_checkpoint_interval_dict = block_checkpoint_interval
    actual_checkpoint_interval = block_checkpoint_interval_dict.get(layer_block_id, checkpoint_interval)
    
    is_checkpoint_saved = (step_block_id % actual_checkpoint_interval == 0)
    is_activation_saved = (layer_block_id % activation_interval == 0)
    
    needs_params = not is_checkpoint_saved and not has_checkpoint
    needs_activation = not is_activation_saved and not has_activation
    
    is_inference = (learning_rate is None)
    
    source_step_block_id = (step_block_id // checkpoint_interval) * checkpoint_interval
    source_step = source_step_block_id * steps_per_block
    
    if layer_block_id > 0:
        prev_layer_block_id = layer_block_id - 1
        prev_layer_names = block_to_layers[prev_layer_block_id]
        prev_last_layer = prev_layer_names[-1]
        
        prev_block_dir = os.path.join(recorder.save_dir, f"block_{prev_layer_block_id}_{step_block_id}")
        prev_layer_dir = os.path.join(prev_block_dir, prev_last_layer)
        prev_step_dir = os.path.join(prev_layer_dir, 'step_0')
        prev_output_path = os.path.join(prev_step_dir, 'output.pt')
        
        if not (os.path.exists(prev_output_path) or os.path.exists(prev_output_path + '.zst')):
            if needs_activation or is_inference:
                
                prev_result = prepare_checkpoint_for_block(
                    prev_layer_block_id, step_block_id, recorder, device, block_records_dir
                )
                
                if 'error' in prev_result:
                    return prev_result
                
                if not (os.path.exists(prev_output_path) or os.path.exists(prev_output_path + '.zst')):
                    return {
                        'error': f'After generating Layer Block {prev_layer_block_id} x Step Block {step_block_id}, still cannot find output file',
                        'required_layer_block': prev_layer_block_id,
                        'required_step_block': step_block_id
                    }
                
    
    
    time_stats = {
        'load_data': 0.0,
        'hash_verification': 0.0,
        'forward_pass': 0.0,
        'backward_pass': 0.0,
        'optimizer_step': 0.0,
        'save_data': 0.0
    }
    
    load_start = time.time()
    layers = {}
    all_params = []
    for layer_name in layer_names:
        layer_module = recorder.load_module_structure(layer_name, device=device)
        layers[layer_name] = layer_module
        if not is_inference:
            all_params.extend(layer_module.parameters())
    
    optimizer = None
    if not is_inference:
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(all_params, lr=learning_rate)
        else:
            optimizer = optim.AdamW(all_params, lr=learning_rate)
    
    if is_inference and block_records_dir:
        final_dir = os.path.join(block_records_dir, 'final_model')
        if not os.path.exists(final_dir):
            return {'error': f'Cannot find final_model directory: {final_dir}'}
        
        for layer_name in layer_names:
            layer_has_params = len(list(layers[layer_name].parameters())) > 0
            if not layer_has_params:
                continue
            
            layer_dir = os.path.join(final_dir, layer_name)
            params_dir = os.path.join(layer_dir, 'parameters')
            
            if not os.path.exists(params_dir):
                return {'error': f'Cannot find {layer_name} parameters directory: {params_dir}'}
            
            with open(os.path.join(params_dir, 'name_mapping.json'), 'r') as f:
                name_mapping = json.load(f)
            
            for safe_name, original_name in name_mapping.items():
                param_file = os.path.join(params_dir, f'{safe_name}.pt')
                param_tensor = torch.load(param_file, map_location=device, weights_only=False)
                
                for name, param in layers[layer_name].named_parameters():
                    if name == original_name:
                        param.data.copy_(param_tensor)
                        break
    else:
        for layer_name in layer_names:
            layer_record = recorder.load_layer_record(layer_name, source_step, device=device)
            
            layer_has_params = len(list(layers[layer_name].parameters())) > 0
            
            if 'parameters' not in layer_record:
                if layer_has_params:
                    return {'error': f'{layer_name} has no parameter record at step {source_step} (may need to generate Step Block {source_step_block_id} checkpoint first)'}
                continue
            
            for param_name, param_tensor in layer_record['parameters'].items():
                for name, param in layers[layer_name].named_parameters():
                    if name == param_name:
                        param.data.copy_(param_tensor)
                        break
            
            if not is_inference and 'optimizer_state' in layer_record:
                for param_name, state_dict in layer_record['optimizer_state'].items():
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
    time_stats['load_data'] += time.time() - load_start
    
    replay_start = source_step
    replay_end = target_step + steps_per_block if needs_activation else target_step
    
    if replay_start < replay_end or (is_inference and needs_activation):
        is_last_block = (layer_names[-1] == layer_order[-1])
        
        pbar = tqdm(range(replay_start, replay_end), desc=f"GPU Replay", unit="step")
        
        all_verification_results = {}
        
        for step in pbar:
            forward_start = time.time()
            layer_outputs = {}
            layer_inputs = {}
            
            is_block_first_layer = {layer_name: (layer_names.index(layer_name) == 0) for layer_name in layer_names}
            is_global_first_layer = {layer_name: (layer_order.index(layer_name) == 0) for layer_name in layer_names}
            
            for idx, layer_name in enumerate(layer_names):
                layer_module = layers[layer_name]
                
                load_start = time.time()
                if layer_name == "embedding":
                    ds_record = recorder.load_dataset_record(step, device=device)
                    layer_input = ds_record['input_ids'].to(device)
                elif is_block_first_layer[layer_name]:
                    input_tensor = recorder.load_layer_input(layer_name, step, device=device)
                    if isinstance(input_tensor, tuple):
                        input_tensor = input_tensor[0]
                    layer_input = input_tensor
                else:
                    prev_layer_name = layer_names[idx - 1]
                    layer_input = layer_outputs[prev_layer_name]
                    if isinstance(layer_input, tuple):
                        layer_input = layer_input[0]
                time_stats['load_data'] += time.time() - load_start
                
                if layer_name == "embedding":
                    if layer_input.dtype == torch.long:
                        computed_output = layer_module(layer_input)
                    else:
                        computed_output = layer_input
                    layer_inputs[layer_name] = layer_input
                elif layer_name.startswith("layer_"):
                    if is_inference:
                        position_ids = torch.arange(0, layer_input.shape[1], device=device).unsqueeze(0)
                        output = layer_module(layer_input, position_ids=position_ids)
                        computed_output = output[0] if isinstance(output, tuple) else output
                        layer_inputs[layer_name] = layer_input
                    else:
                        input_grad_tensor = layer_input.clone().requires_grad_(True)
                        input_grad_tensor.retain_grad()
                        position_ids = torch.arange(0, input_grad_tensor.shape[1], device=device).unsqueeze(0)
                        output = layer_module(input_grad_tensor, position_ids=position_ids)
                        computed_output = output[0] if isinstance(output, tuple) else output
                        layer_inputs[layer_name] = input_grad_tensor
                elif layer_name in ["norm", "lm_head"]:
                    if is_inference:
                        computed_output = layer_module(layer_input)
                        layer_inputs[layer_name] = layer_input
                    else:
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
                    
                    if is_inference:
                        computed_output = layer_module(processed_input)
                        layer_inputs[layer_name] = processed_input
                    else:
                        input_grad_tensor = processed_input.clone().requires_grad_(True)
                        input_grad_tensor.retain_grad()
                        computed_output = layer_module(input_grad_tensor)
                        layer_inputs[layer_name] = input_grad_tensor
                else:
                    if is_inference:
                        computed_output = layer_module(layer_input)
                        layer_inputs[layer_name] = layer_input
                    else:
                        input_grad_tensor = layer_input.clone().requires_grad_(True)
                        input_grad_tensor.retain_grad()
                        computed_output = layer_module(input_grad_tensor)
                        layer_inputs[layer_name] = input_grad_tensor
                
                layer_outputs[layer_name] = computed_output
            time_stats['forward_pass'] += time.time() - forward_start
            
            if not is_inference:
                backward_start = time.time()
                optimizer.zero_grad()
                
                last_layer = layer_names[-1]
                
                load_start = time.time()
                if not is_last_block:
                    expected_output_grad = recorder.load_layer_output_grad(last_layer, step, device=device)
                    last_output = layer_outputs[last_layer]
                    if isinstance(last_output, tuple):
                        last_output = last_output[0]
                    last_output.backward(expected_output_grad, retain_graph=True)
                else:
                    if last_layer == 'lm_head':
                        ds_record = recorder.load_dataset_record(step, device=device)
                        labels = ds_record['labels'].to(device)
                        logits = layer_outputs['lm_head']
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        logits = logits.to(torch.float32)
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss()
                        computed_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        computed_loss.backward()
                    else:
                        expected_output_grad = recorder.load_layer_output_grad(last_layer, step, device=device)
                        last_output = layer_outputs[last_layer]
                        if isinstance(last_output, tuple):
                            last_output = last_output[0]
                        last_output.backward(expected_output_grad, retain_graph=True)
                time_stats['load_data'] += time.time() - load_start
                
                opt_start = time.time()
                optimizer.step()
                time_stats['optimizer_step'] += time.time() - opt_start
                time_stats['backward_pass'] += time.time() - backward_start
            
            if step >= target_step and step < target_step + steps_per_block and needs_activation:
                save_start = time.time()
                step_in_block = step % steps_per_block
                verification_result = save_step_activations_and_grads(
                    recorder, layer_names, layer_block_id, step_block_id,
                    step_in_block, layer_inputs, layer_outputs, layer_order
                )
                all_verification_results[step] = verification_result
                time_stats['save_data'] += time.time() - save_start
        
        pbar.close()
    
    if needs_params:
        hash_start = time.time()
        param_verification_passed = True
        param_mismatches = []
        
        target_block_dir = os.path.join(recorder.save_dir, f"block_{layer_block_idx}_{step_block_id}")
        
        for layer_name in layer_names:
            layer_dir = os.path.join(target_block_dir, layer_name)
            checkpoint_hash_file = os.path.join(layer_dir, 'checkpoint_hashes.json')
            
            if os.path.exists(checkpoint_hash_file):
                with open(checkpoint_hash_file, 'r') as f:
                    checkpoint_hashes = json.load(f)
                    expected_param_hashes = checkpoint_hashes.get('parameter_hashes', {})
                
                layer_module = layers[layer_name]
                for name, param in layer_module.named_parameters():
                    expected_hash = expected_param_hashes.get(name, '')
                    if expected_hash:
                        actual_hash = get_hash(param, recorder.chunk_size, recorder.use_blake3)
                        if actual_hash != expected_hash:
                            param_verification_passed = False
                            param_mismatches.append((layer_name, name, expected_hash, actual_hash))
        
        time_stats['hash_verification'] += time.time() - hash_start
        
        if not param_verification_passed:
            return {
                'error': 'Parameters hash verification failed',
                'param_mismatches': param_mismatches
            }
        
        
        save_start = time.time()
        os.makedirs(target_block_dir, exist_ok=True)
        
        for layer_name in layer_names:
            layer_dir = os.path.join(target_block_dir, layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            
            layer_module = layers[layer_name]
            
            params_dir = os.path.join(layer_dir, 'parameters')
            os.makedirs(params_dir, exist_ok=True)
            
            param_name_mapping = {}
            for idx, (name, param) in enumerate(layer_module.named_parameters()):
                safe_name = f'param_{idx}'
                param_name_mapping[safe_name] = name
                param_path = os.path.join(params_dir, f'{safe_name}.pt')
                torch.save(param.detach().cpu(), param_path)
            
            with open(os.path.join(params_dir, 'name_mapping.json'), 'w') as f:
                json.dump(param_name_mapping, f, indent=2)
            
            optimizer_state_dir = os.path.join(layer_dir, 'optimizer_state')
            os.makedirs(optimizer_state_dir, exist_ok=True)
            
            opt_name_mapping = {}
            opt_idx = 0
            for name, param in layer_module.named_parameters():
                if param in optimizer.state:
                    state = optimizer.state[param]
                    state_dict = {}
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state_dict[k] = v.detach().cpu()
                        else:
                            state_dict[k] = v
                    
                    if state_dict:
                        safe_name = f'opt_{opt_idx}'
                        opt_name_mapping[safe_name] = name
                        state_path = os.path.join(optimizer_state_dir, f'{safe_name}.pt')
                        torch.save(state_dict, state_path)
                        opt_idx += 1
            
            if opt_name_mapping:
                with open(os.path.join(optimizer_state_dir, 'name_mapping.json'), 'w') as f:
                    json.dump(opt_name_mapping, f, indent=2)
            
            checkpoint_hash_file = os.path.join(layer_dir, 'checkpoint_hashes.json')
            
            if os.path.exists(checkpoint_hash_file):
                with open(checkpoint_hash_file, 'r') as f:
                    checkpoint_hashes = json.load(f)
            else:
                checkpoint_hashes = {}
            
            checkpoint_hashes['has_checkpoint'] = True
            checkpoint_hashes['generated_from_step_block'] = source_step_block_id
            
            with open(checkpoint_hash_file, 'w') as f:
                json.dump(checkpoint_hashes, f, indent=2)
        time_stats['save_data'] += time.time() - save_start
        
    
    total_checks = 0
    passed_checks = 0
    failed_checks = 0
    
    if needs_activation:
        
        if all_verification_results:
            for step, step_results in sorted(all_verification_results.items()):
                for layer_name, layer_results in step_results.items():
                    for data_type, (is_match, expected_hash, actual_hash) in layer_results.items():
                        total_checks += 1
                        if is_match:
                            passed_checks += 1
                        else:
                            failed_checks += 1
    
    return {
        'status': 'success',
        'source_step_block': source_step_block_id,
        'target_step_block': step_block_id,
        'layer_names': layer_names,
        'verification_passed': (failed_checks == 0) if needs_activation else None,
        'total_checks': total_checks if needs_activation else 0,
        'passed_checks': passed_checks if needs_activation else 0,
        'failed_checks': failed_checks if needs_activation else 0,
        'time_stats': time_stats
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', type=str, default='./block_records',
                       help='Record directory (training: ./block_records, inference: ./inference_records/<prompt_hash>)')
    parser.add_argument('--block_records_dir', type=str, default='./block_records',
                       help='Training record directory (used for loading parameters in inference, default: ./block_records)')
    parser.add_argument('--layer_block_id', type=int, required=True, help='Layer Block ID')
    parser.add_argument('--step_block_id', type=int, required=True, help='Step Block ID')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_deterministic(seed=args.seed, enabled=args.deterministic)
    
    temp_recorder = LayerRecorder(save_dir=args.record_dir)
    metadata = temp_recorder.load_metadata()
    
    is_inference = (metadata['learning_rate'] is None)
    
    if is_inference:
        block_recorder = LayerRecorder(save_dir=args.block_records_dir)
        block_metadata = block_recorder.load_metadata()
        module_structure_dir = block_metadata['module_structure_dir'] if 'module_structure_dir' in block_metadata else metadata['module_structure_dir']
        model_name = block_metadata['model_name'] if 'model_name' in block_metadata else metadata['model_name']
    else:
        module_structure_dir = metadata['module_structure_dir']
        model_name = metadata['model_name']
    
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
        module_structure_dir=module_structure_dir,
        model_name=model_name
    )
    
    recorder.layer_to_block = metadata['layer_to_block']
    recorder.block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
    recorder.layer_order = metadata['layer_order']
    recorder.layer_order_finalized = True
    recorder.layer_blocks_finalized = True
    
    checkpoint_interval = metadata['checkpoint_interval']
    activation_interval = metadata['activation_interval']
    if checkpoint_interval > 1 and activation_interval > 1:
        raise ValueError("checkpoint_interval and activation_interval cannot both be greater than 1")
    
    result = prepare_checkpoint_for_block(
        args.layer_block_id,
        args.step_block_id,
        recorder,
        args.device,
        args.block_records_dir if is_inference else None
    )
    
    if 'error' in result:
        raise RuntimeError(result['error'])
    
    if result['status'] == 'already_exists':
        pass
    elif result['status'] == 'success':
        
        if 'time_stats' in result:
            time_stats = result['time_stats']
            total_time = sum(time_stats.values())
            print("\n" + "="*60)
            print("Generation time statistics")
            print("="*60)
            print(f"Data loading:         {time_stats['load_data']:7.3f} seconds ({time_stats['load_data']/total_time*100:5.1f}%)")
            print(f"Hash verification:    {time_stats['hash_verification']:7.3f} seconds ({time_stats['hash_verification']/total_time*100:5.1f}%)")
            print(f"Forward computation:  {time_stats['forward_pass']:7.3f} seconds ({time_stats['forward_pass']/total_time*100:5.1f}%)")
            print(f"Backward computation: {time_stats['backward_pass']:7.3f} seconds ({time_stats['backward_pass']/total_time*100:5.1f}%)")
            print(f"Optimizer update:     {time_stats['optimizer_step']:7.3f} seconds ({time_stats['optimizer_step']/total_time*100:5.1f}%)")
            print(f"Data saving:          {time_stats['save_data']:7.3f} seconds ({time_stats['save_data']/total_time*100:5.1f}%)")
            print(f"Total time:           {total_time:7.3f} seconds")
            print("="*60)


if __name__ == '__main__':
    main()

