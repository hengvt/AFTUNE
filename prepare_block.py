import torch
import torch.optim as optim
import argparse
import os
import json
import time
from tqdm import tqdm
from layer_recorder import LayerRecorder, set_deterministic
from verify_block import (
    copy_module_params,
    forward_block_layer,
    get_hash,
    lm_head_cross_entropy_loss,
    restore_optimizer_state,
)
from decoder_replay import load_saved_rotary_emb, new_rope_state


# Build replay modules from structures; load params from checkpoint or final_model
def setup_layers_and_optimizer(recorder, layer_names, device, is_inference, optimizer_type, learning_rate,
                               block_records_dir, source_step, source_step_block_id):
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
            return None, None, {'error': f'Cannot find final_model directory: {final_dir}'}
        for layer_name in layer_names:
            layer_module = layers[layer_name]
            if len(list(layer_module.parameters())) == 0:
                continue
            params_dir = os.path.join(final_dir, layer_name, 'parameters')
            if not os.path.exists(params_dir):
                return None, None, {'error': f'Cannot find {layer_name} parameters directory: {params_dir}'}
            with open(os.path.join(params_dir, 'name_mapping.json'), 'r') as f:
                name_mapping = json.load(f)
            param_tensors = {}
            for safe_name, original_name in name_mapping.items():
                param_file = os.path.join(params_dir, f'{safe_name}.pt')
                param_tensors[original_name] = torch.load(param_file, map_location=device, weights_only=False)
            copy_module_params(layer_module, param_tensors)
    else:
        for layer_name in layer_names:
            layer_module = layers[layer_name]
            layer_record = recorder.load_layer_record(layer_name, source_step, device=device)
            if 'parameters' not in layer_record:
                if len(list(layer_module.parameters())) > 0:
                    return None, None, {
                        'error': f'{layer_name} has no parameter record at step {source_step} (may need to generate Step Block {source_step_block_id} checkpoint first)',
                    }
                continue
            copy_module_params(layer_module, layer_record['parameters'])
            if not is_inference and 'optimizer_state' in layer_record:
                restore_optimizer_state(optimizer, layer_module, layer_record, device)

    return layers, optimizer, None


# Persist params + optimizer state under block_{layer}_{step}/ for later replay
def save_block_checkpoint(layers, layer_names, optimizer, target_block_dir, source_step_block_id):
    os.makedirs(target_block_dir, exist_ok=True)
    for layer_name in layer_names:
        layer_dir = os.path.join(target_block_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        layer_module = layers[layer_name]

        params_dir = os.path.join(layer_dir, 'parameters')
        os.makedirs(params_dir, exist_ok=True)
        param_name_mapping = {}
        for param_idx, (name, param) in enumerate(layer_module.named_parameters()):
            safe_name = f'param_{param_idx}'
            param_name_mapping[safe_name] = name
            torch.save(param.detach().cpu(), os.path.join(params_dir, f'{safe_name}.pt'))
        with open(os.path.join(params_dir, 'name_mapping.json'), 'w') as f:
            json.dump(param_name_mapping, f, indent=2)

        optimizer_state_dir = os.path.join(layer_dir, 'optimizer_state')
        os.makedirs(optimizer_state_dir, exist_ok=True)
        opt_name_mapping = {}
        opt_idx = 0
        for name, param in layer_module.named_parameters():
            if param not in optimizer.state:
                continue
            state_dict = {}
            for k, v in optimizer.state[param].items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v.detach().cpu()
                else:
                    state_dict[k] = v
            if not state_dict:
                continue
            safe_name = f'opt_{opt_idx}'
            opt_name_mapping[safe_name] = name
            torch.save(state_dict, os.path.join(optimizer_state_dir, f'{safe_name}.pt'))
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


# Write per-step I/O tensors to block dir and match recorded hashes
def save_step_activations_and_grads(recorder, layer_names, layer_block_id, step_block_id,
                                     step_in_block, layer_inputs, layer_outputs, layer_order):
    block_dir = os.path.join(recorder.save_dir, f"block_{layer_block_id}_{step_block_id}")
    os.makedirs(block_dir, exist_ok=True)
    is_global_first = (layer_names[0] == layer_order[0])
    verification_results = {}

    for layer_name in layer_names:
        step_dir = os.path.join(block_dir, layer_name, f"step_{step_in_block}")
        os.makedirs(step_dir, exist_ok=True)
        verification_results[layer_name] = {}

        expected_hashes = {}
        hash_file = os.path.join(step_dir, 'hashes.json')
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                expected_hashes = json.load(f)

        def save_and_verify(tensor, filename, hash_key):
            if tensor is None:
                return
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            recorder.save_tensor_compressed(tensor.detach().cpu(), os.path.join(step_dir, filename))
            actual_hash = get_hash(tensor, recorder.chunk_size, recorder.use_blake3)
            expected_hash = expected_hashes[hash_key] if hash_key in expected_hashes else ''
            verification_results[layer_name][hash_key.replace('_hash', '')] = (
                actual_hash == expected_hash, expected_hash, actual_hash
            )

        is_block_first = (layer_name == layer_names[0])
        is_block_last = (layer_name == layer_names[-1])
        is_global_last = (layer_name == layer_order[-1])

        if is_global_first and is_block_first and layer_name in layer_inputs:
            save_and_verify(layer_inputs[layer_name], 'input.pt', 'input_hash')
        if is_block_last and layer_name in layer_outputs:
            save_and_verify(layer_outputs[layer_name], 'output.pt', 'output_hash')

        input_tensor = layer_inputs[layer_name] if layer_name in layer_inputs else None
        output_tensor = layer_outputs[layer_name] if layer_name in layer_outputs else None
        if is_block_first and input_tensor is not None and hasattr(input_tensor, 'grad') and input_tensor.grad is not None:
            save_and_verify(input_tensor.grad, 'input_grad.pt', 'input_grad_hash')
        if is_global_last and output_tensor is not None and hasattr(output_tensor, 'grad') and output_tensor.grad is not None:
            save_and_verify(output_tensor.grad, 'output_grad.pt', 'output_grad_hash')

    return verification_results


# GPU replay to produce missing block checkpoint and/or activation files
def prepare_checkpoint_for_block(layer_block_id, step_block_id, recorder, device, block_records_dir):
    metadata = recorder.load_metadata()
    steps_per_block = metadata['steps_per_block']
    checkpoint_interval = metadata['checkpoint_interval']
    activation_interval = metadata['activation_interval']
    layer_order = metadata['layer_order']
    optimizer_type = metadata['optimizer_type']
    learning_rate = metadata['learning_rate']
    block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
    model_name = metadata['model_name']
    block_checkpoint_interval = {int(k): v for k, v in metadata['block_checkpoint_interval'].items()}

    if layer_block_id not in block_to_layers:
        return {'error': f'Layer Block {layer_block_id} does not exist'}

    layer_names = block_to_layers[layer_block_id]
    target_step = step_block_id * steps_per_block
    block_dir = os.path.join(recorder.save_dir, f"block_{layer_block_id}_{step_block_id}")

    has_checkpoint = False
    has_activation = False
    if os.path.exists(block_dir):
        checkpoint_hash_file = os.path.join(block_dir, layer_names[0], 'checkpoint_hashes.json')
        if os.path.exists(checkpoint_hash_file):
            with open(checkpoint_hash_file, 'r') as f:
                checkpoint_info = json.load(f)
            if 'has_checkpoint' in checkpoint_info:
                has_checkpoint = checkpoint_info['has_checkpoint']
        step_dir = os.path.join(block_dir, layer_names[-1], 'step_0')
        has_activation = (
            recorder.file_exists(os.path.join(step_dir, 'output.pt'))
            or recorder.file_exists(os.path.join(step_dir, 'input.pt'))
        )

    if layer_block_id in block_checkpoint_interval:
        actual_checkpoint_interval = block_checkpoint_interval[layer_block_id]
    else:
        actual_checkpoint_interval = checkpoint_interval

    needs_params = not (step_block_id % actual_checkpoint_interval == 0) and not has_checkpoint
    needs_activation = not (layer_block_id % activation_interval == 0) and not has_activation
    is_inference = (learning_rate is None)
    source_step_block_id = (step_block_id // checkpoint_interval) * checkpoint_interval
    source_step = source_step_block_id * steps_per_block

    # Ensure previous layer block output exists before replaying this block
    if layer_block_id > 0:
        prev_layer_block_id = layer_block_id - 1
        prev_output_path = os.path.join(
            recorder.save_dir, f"block_{prev_layer_block_id}_{step_block_id}",
            block_to_layers[prev_layer_block_id][-1], 'step_0', 'output.pt',
        )
        if not recorder.file_exists(prev_output_path) and (needs_activation or is_inference):
            prev_result = prepare_checkpoint_for_block(
                prev_layer_block_id, step_block_id, recorder, device, block_records_dir,
            )
            if 'error' in prev_result:
                return prev_result
            if not recorder.file_exists(prev_output_path):
                return {
                    'error': f'After generating Layer Block {prev_layer_block_id} x Step Block {step_block_id}, still cannot find output file',
                    'required_layer_block': prev_layer_block_id,
                    'required_step_block': step_block_id,
                }

    time_stats = {k: 0.0 for k in ('load_data', 'hash_verification', 'forward_pass', 'backward_pass', 'optimizer_step', 'save_data')}

    load_start = time.time()
    layers, optimizer, setup_error = setup_layers_and_optimizer(
        recorder, layer_names, device, is_inference, optimizer_type, learning_rate,
        block_records_dir, source_step, source_step_block_id,
    )
    if setup_error is not None:
        return setup_error
    time_stats['load_data'] += time.time() - load_start

    replay_start = source_step
    replay_end = target_step + steps_per_block if needs_activation else target_step
    all_verification_results = {}

    if replay_start < replay_end or (is_inference and needs_activation):
        is_last_block = (layer_names[-1] == layer_order[-1])
        track_grad = not is_inference
        rotary_emb_module = load_saved_rotary_emb(recorder, model_name, device)
        rope_state = new_rope_state(rotary_emb_module, model_name)
        rotary_emb_cache = {}

        # Train-style replay: forward, backward, optimizer.step when not inference
        pbar = tqdm(range(replay_start, replay_end), desc=f"GPU Replay", unit="step")
        for step in pbar:
            forward_start = time.time()
            layer_outputs = {}
            layer_inputs = {}
            if rope_state is not None:
                rope_state["text_pos"] = None
                rope_state["embeddings"] = None

            for idx, layer_name in enumerate(layer_names):
                layer_module = layers[layer_name]

                load_start = time.time()
                if layer_name == "embedding":
                    ds_record = recorder.load_dataset_record(step, device=device)
                    layer_input = ds_record['input_ids'].to(device)
                elif idx == 0:
                    layer_input = recorder.load_layer_input(layer_name, step, device=device)
                else:
                    layer_input = layer_outputs[layer_names[idx - 1]]
                    if isinstance(layer_input, tuple):
                        layer_input = layer_input[0]
                time_stats['load_data'] += time.time() - load_start

                computed_output, layer_inputs[layer_name] = forward_block_layer(
                    layer_name, layer_module, layer_input, model_name, track_grad,
                    rotary_emb_module, rotary_emb_cache, rope_state,
                )
                if isinstance(computed_output, tuple):
                    computed_output = computed_output[0]
                layer_outputs[layer_name] = computed_output
            time_stats['forward_pass'] += time.time() - forward_start

            if not is_inference:
                backward_start = time.time()
                load_start = time.time()
                optimizer.zero_grad()
                last_layer = layer_names[-1]
                if is_last_block and last_layer == 'lm_head':
                    ds_record = recorder.load_dataset_record(step, device=device)
                    lm_head_cross_entropy_loss(layer_outputs['lm_head'], ds_record['labels'], device).backward()
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
                all_verification_results[step] = save_step_activations_and_grads(
                    recorder, layer_names, layer_block_id, step_block_id,
                    step % steps_per_block, layer_inputs, layer_outputs, layer_order,
                )
                time_stats['save_data'] += time.time() - save_start
        pbar.close()

    if needs_params:
        target_block_dir = os.path.join(recorder.save_dir, f"block_{layer_block_id}_{step_block_id}")
        hash_start = time.time()
        param_mismatches = []
        for layer_name in layer_names:
            checkpoint_hash_file = os.path.join(target_block_dir, layer_name, 'checkpoint_hashes.json')
            if not os.path.exists(checkpoint_hash_file):
                continue
            with open(checkpoint_hash_file, 'r') as f:
                checkpoint_hashes = json.load(f)
            if 'parameter_hashes' not in checkpoint_hashes:
                continue
            expected_param_hashes = checkpoint_hashes['parameter_hashes']
            for name, param in layers[layer_name].named_parameters():
                if name not in expected_param_hashes:
                    continue
                expected_hash = expected_param_hashes[name]
                if not expected_hash:
                    continue
                actual_hash = get_hash(param, recorder.chunk_size, recorder.use_blake3)
                if actual_hash != expected_hash:
                    param_mismatches.append((layer_name, name, expected_hash, actual_hash))
        time_stats['hash_verification'] += time.time() - hash_start
        if param_mismatches:
            return {'error': 'Parameters hash verification failed', 'param_mismatches': param_mismatches}

        save_start = time.time()
        save_block_checkpoint(layers, layer_names, optimizer, target_block_dir, source_step_block_id)
        time_stats['save_data'] += time.time() - save_start

    total_checks = 0
    passed_checks = 0
    if needs_activation and all_verification_results:
        for step_results in all_verification_results.values():
            for layer_results in step_results.values():
                for is_match, _, _ in layer_results.values():
                    total_checks += 1
                    if is_match:
                        passed_checks += 1
    failed_checks = total_checks - passed_checks

    return {
        'status': 'success',
        'source_step_block': source_step_block_id,
        'target_step_block': step_block_id,
        'layer_names': layer_names,
        'verification_passed': (failed_checks == 0) if needs_activation else None,
        'total_checks': total_checks if needs_activation else 0,
        'passed_checks': passed_checks if needs_activation else 0,
        'failed_checks': failed_checks if needs_activation else 0,
        'time_stats': time_stats,
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
        model_name=model_name,
    )
    recorder.layer_to_block = metadata['layer_to_block']
    recorder.block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
    recorder.layer_order = metadata['layer_order']
    recorder.layer_order_finalized = True
    recorder.layer_blocks_finalized = True

    if metadata['checkpoint_interval'] > 1 and metadata['activation_interval'] > 1:
        raise ValueError("checkpoint_interval and activation_interval cannot both be greater than 1")

    result = prepare_checkpoint_for_block(
        args.layer_block_id, args.step_block_id, recorder, args.device,
        args.block_records_dir if is_inference else None,
    )
    if 'error' in result:
        raise RuntimeError(result['error'])

    if result['status'] == 'success' and 'time_stats' in result:
        time_stats = result['time_stats']
        total_time = sum(time_stats.values())
        print("\n" + "=" * 60)
        print("Generation time statistics")
        print("=" * 60)
        for label, key in (
            ("Data loading", 'load_data'),
            ("Hash verification", 'hash_verification'),
            ("Forward computation", 'forward_pass'),
            ("Backward computation", 'backward_pass'),
            ("Optimizer update", 'optimizer_step'),
            ("Data saving", 'save_data'),
        ):
            t = time_stats[key]
            print(f"{label + ':':22} {t:7.3f} seconds ({t / total_time * 100:5.1f}%)")
        print(f"{'Total time:':22} {total_time:7.3f} seconds")
        print("=" * 60)


if __name__ == '__main__':
    main()
