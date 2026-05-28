import torch
import torch.optim as optim
import argparse
import os
import json
import time
from tqdm import tqdm
from layer_recorder import LayerRecorder, set_deterministic
from decoder_replay import load_saved_rotary_emb, new_rope_state, replay_decoder_layer
import aftune_torch


# Merkle-root hash of a tensor (matches layer_recorder hashing)
def get_hash(tensor, chunk_size, use_blake3):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    root = aftune_torch.ops.flat_root(tensor, chunk_size=chunk_size, use_blake3=use_blake3)
    return root.hex()


# Single-layer forward used by verify / prepare / verify_inference replay
def forward_block_layer(layer_name, layer_module, layer_input, model_name, track_grad,
                        rotary_emb_module, rotary_emb_cache, rope_state):
    if layer_name == "embedding":
        if layer_input.dtype == torch.long:
            output = layer_module(layer_input)
        else:
            output = layer_input
        return output, layer_input
    if layer_name == "classifier":
        if model_name == "dinov2_giant" and len(layer_input.shape) == 3:
            cls_token = layer_input[:, 0]
            patch_tokens = layer_input[:, 1:]
            patch_mean = patch_tokens.mean(dim=1)
            layer_input = torch.cat([cls_token, patch_mean], dim=1)
        elif model_name == "vit_large" and len(layer_input.shape) == 3:
            layer_input = layer_input[:, 0, :]
    if layer_name.startswith("layer_"):
        if track_grad:
            x = layer_input.clone().requires_grad_(True)
            x.retain_grad()
            output = replay_decoder_layer(layer_module, x, rotary_emb_module, rotary_emb_cache, rope_state)
            return output, x
        return replay_decoder_layer(layer_module, layer_input, rotary_emb_module, rotary_emb_cache, rope_state), layer_input
    if track_grad:
        x = layer_input.clone().requires_grad_(True)
        x.retain_grad()
        return layer_module(x), x
    return layer_module(layer_input), layer_input


# Copy saved tensors into module parameters for replay
def copy_module_params(module, param_tensors):
    for name, param in module.named_parameters():
        if name in param_tensors:
            param.data.copy_(param_tensors[name])


# Rebuild optimizer.state from a layer record at step-block boundary
def restore_optimizer_state(optimizer, layer_module, layer_record, device):
    for param_name, state_dict in layer_record['optimizer_state'].items():
        for name, param in layer_module.named_parameters():
            if name != param_name:
                continue
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


# Causal LM loss for lm_head backward during replay
def lm_head_cross_entropy_loss(logits, labels, device):
    labels = labels.to(device)
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits.to(torch.float32)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


# Compare replayed tensor to recorded reference (hash or numeric L2 stats)
def verify_tensor_diff(actual, expected, key, expected_hash, strict_hash, chunk_size, use_blake3):
    if strict_hash:
        replayed_hash = get_hash(actual, chunk_size, use_blake3)
        if replayed_hash != expected_hash:
            return {'hash_match': False}, f'{key} hash mismatch after replay (not bitwise identical)'
        return {'hash_match': True}, None
    compare_chunk = 1024 * 1024
    total_elements = actual.numel()
    sum_diff_sq = 0.0
    sum_expected_sq = 0.0
    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    for i in range(0, total_elements, compare_chunk):
        end_idx = min(i + compare_chunk, total_elements)
        diff_chunk = (actual_flat[i:end_idx] - expected_flat[i:end_idx]).abs()
        sum_diff_sq += torch.sum(diff_chunk * diff_chunk).item()
        sum_expected_sq += torch.sum(expected_flat[i:end_idx] * expected_flat[i:end_idx]).item()
    return {
        'diff_norm_sq': sum_diff_sq,
        'expected_norm_sq': sum_expected_sq,
    }, None


# Load final_model/ weights and hashes for end-of-training comparison
def load_final_model_records_for_compare(recorder, layer_names, device):
    records_for_compare = {}
    final_dir = os.path.join(recorder.save_dir, 'final_model')
    for layer_name in layer_names:
        layer_dir = os.path.join(final_dir, layer_name)
        if not os.path.exists(layer_dir):
            continue
        records_for_compare[layer_name] = {'parameters': {}, 'parameter_hashes': {}}
        params_dir = os.path.join(layer_dir, 'parameters')
        if os.path.exists(params_dir):
            with open(os.path.join(params_dir, 'name_mapping.json'), 'r') as f:
                name_mapping = json.load(f)
            for safe_name, original_name in name_mapping.items():
                p = os.path.join(params_dir, f'{safe_name}.pt')
                if os.path.exists(p):
                    records_for_compare[layer_name]['parameters'][original_name] = torch.load(
                        p, map_location=device, weights_only=False,
                    )
        hash_file = os.path.join(layer_dir, 'parameter_hashes.json')
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                records_for_compare[layer_name]['parameter_hashes'] = json.load(f)
    return records_for_compare


def verify_block(layer_block_id, step_block_id, recorder, device, strict_hash):
    # Replay one layer block × step block on GPU and check hashes / weight drift
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
    use_lora = metadata['use_lora']

    if layer_block_id not in block_to_layers:
        return {'error': f'Layer Block {layer_block_id} does not exist'}

    layer_names = block_to_layers[layer_block_id]
    is_last_block = ('lm_head' in layer_names)
    global_first_layer = layer_order[0]

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
        'weight_comparison': 0.0,
    }

    block_start_step = step_block_id * steps_per_block
    block_end_step = (step_block_id + 1) * steps_per_block
    if total_steps is not None:
        block_end_step = min(block_end_step, total_steps)
        if block_start_step >= total_steps:
            return {'error': f'Step Block {step_block_id} exceeds training range (total_steps={total_steps})'}

    # Load checkpoint params + optimizer state at step-block start; verify disk hashes
    load_start = time.time()
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
        del first_layer_record
        return {'error': f'{layer_names[0]} has no parameter record at step {block_start_step}'}
    del first_layer_record

    for layer_name in layer_names:
        layer_record = recorder.load_layer_record(layer_name, block_start_step, device=device)
        layer_has_params = len(list(layers[layer_name].parameters())) > 0

        if 'parameters' not in layer_record:
            if layer_has_params and not use_lora:
                return {'error': f'{layer_name} has no parameter record at step {block_start_step}'}
        else:
            hash_start = time.time()
            for param_name in layer_record['parameter_hashes']:
                if param_name not in layer_record['parameters']:
                    continue
                param_tensor = layer_record['parameters'][param_name]
                expected_hash = layer_record['parameter_hashes'][param_name]
                actual_hash = get_hash(param_tensor, chunk_size, use_blake3)
                if actual_hash != expected_hash:
                    return {
                        'error': f'{layer_name}.{param_name} hash mismatch (Step {block_start_step})\n'
                                 f'  Expected hash: {expected_hash}\n'
                                 f'  Actual hash: {actual_hash}\n',
                    }
            copy_module_params(layers[layer_name], layer_record['parameters'])
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
                            return {
                                'error': f'{layer_name}.{hash_key} optimizer state hash mismatch (Step {block_start_step})\n'
                                         f'  Expected hash: {expected_hash}\n'
                                         f'  Actual hash: {actual_hash}\n',
                            }
            restore_optimizer_state(optimizer, layers[layer_name], layer_record, device)
            del layer_record['optimizer_state']
            time_stats['hash_verification'] += time.time() - hash_start

        del layer_record
    time_stats['load_data'] += time.time() - load_start

    rotary_emb_module = load_saved_rotary_emb(recorder, model_name, device)
    rope_state = new_rope_state(rotary_emb_module, model_name)
    rotary_emb_cache = {}

    # Step-by-step forward / backward / optimizer.step replay inside this block
    pbar = tqdm(range(block_start_step, block_end_step), desc=f"Replay Layer Block {layer_block_id} × Step Block {step_block_id}", unit="step")
    for step in pbar:
        load_start = time.time()
        step_records = {name: recorder.load_layer_record(name, step, device=device) for name in layer_names}
        time_stats['load_data'] += time.time() - load_start

        ds_record = recorder.load_dataset_record(step, device=device) if is_last_block else None

        if is_last_block:
            loss_start = time.time()
            if 'loss' in ds_record:
                expected_loss = ds_record['loss']
                logits = step_records['lm_head']['output'].to(device)
                computed_loss = lm_head_cross_entropy_loss(logits, ds_record['labels'], device)
                loss_diff = abs(computed_loss.item() - expected_loss)
                if loss_diff > 1e-3:
                    pbar.close()
                    return {'error': f'Loss verification failed at step {step}: difference {loss_diff:.6e}'}
            time_stats['loss_verification'] += time.time() - loss_start

        layer_outputs = {}
        layer_inputs = {}
        if rope_state is not None:
            rope_state["text_pos"] = None
            rope_state["embeddings"] = None

        for idx, layer_name in enumerate(layer_names):
            step_record = step_records[layer_name]
            layer_module = layers[layer_name]

            if idx == 0:
                hash_start = time.time()
                input_tensor = recorder.load_layer_input(layer_name, step, device=device)
                if layer_name == global_first_layer and 'input_hash' in step_record:
                    actual_input_hash = get_hash(input_tensor, chunk_size, use_blake3)
                    if actual_input_hash != step_record['input_hash']:
                        pbar.close()
                        return {'error': f'{layer_name}: Input hash mismatch (step {step})'}
                time_stats['hash_verification'] += time.time() - hash_start
            else:
                input_tensor = layer_outputs[layer_names[idx - 1]]
                if isinstance(input_tensor, tuple):
                    input_tensor = input_tensor[0]

            forward_start = time.time()
            computed_output, layer_inputs[layer_name] = forward_block_layer(
                layer_name, layer_module, input_tensor, model_name, True,
                rotary_emb_module, rotary_emb_cache, rope_state,
            )
            if isinstance(computed_output, tuple):
                computed_output = computed_output[0]
            layer_outputs[layer_name] = computed_output
            time_stats['forward_pass'] += time.time() - forward_start

            if idx == len(layer_names) - 1:
                hash_start = time.time()
                expected_output = recorder.load_layer_output(layer_name, step, device=device)
                if 'output_hash' in step_record:
                    actual_output_hash = get_hash(expected_output, chunk_size, use_blake3)
                    if actual_output_hash != step_record['output_hash']:
                        pbar.close()
                        return {'error': f'{layer_name}: Output hash mismatch (disk data corrupted, step {step})'}
                time_stats['hash_verification'] += time.time() - hash_start
                del expected_output

        first_layer = layer_names[0]
        last_layer = layer_names[-1]
        backward_start = time.time()
        optimizer.zero_grad()
        if is_last_block and last_layer == 'lm_head':
            lm_head_cross_entropy_loss(layer_outputs['lm_head'], ds_record['labels'], device).backward()
        else:
            hash_start = time.time()
            expected_output_grad = recorder.load_layer_output_grad(last_layer, step, device=device)
            time_stats['hash_verification'] += time.time() - hash_start
            last_output = layer_outputs[last_layer]
            if isinstance(last_output, tuple):
                last_output = last_output[0]
            last_output.backward(expected_output_grad, retain_graph=True)
        time_stats['backward_pass'] += time.time() - backward_start

        if 'input_grad_hash' in step_records[first_layer]:
            hash_start = time.time()
            expected_input_grad = recorder.load_layer_input_grad(first_layer, step, device=device)
            actual_hash = get_hash(expected_input_grad, chunk_size, use_blake3)
            if actual_hash != step_records[first_layer]['input_grad_hash']:
                pbar.close()
                return {'error': f'{first_layer}: Input grad hash mismatch (step {step})'}
            if first_layer in layer_inputs and hasattr(layer_inputs[first_layer], 'grad') and layer_inputs[first_layer].grad is not None:
                del expected_input_grad
            time_stats['hash_verification'] += time.time() - hash_start

        opt_start = time.time()
        optimizer.step()
        time_stats['optimizer_step'] += time.time() - opt_start

        del step_records, layer_outputs, layer_inputs, ds_record

    pbar.close()

    weight_verification = None
    optimizer_state_verification = None
    # After replay: compare weights to next step block or final_model
    is_last_step_block = (total_steps is not None and block_end_step >= total_steps)
    if not is_last_step_block:
        compare_source = 'next_block'
    elif os.path.exists(os.path.join(recorder.save_dir, 'final_model')):
        compare_source = 'final_model'
    else:
        compare_source = None

    records_for_compare = None
    if compare_source == 'next_block':
        load_start = time.time()
        next_block_start = (step_block_id + 1) * steps_per_block
        records_for_compare = {}
        for layer_name in layer_names:
            layer_record = recorder.load_layer_record(layer_name, next_block_start, device=device)
            entry = {
                'parameters': {},
                'parameter_hashes': {},
                'optimizer_state': {},
                'optimizer_state_hashes': {},
            }
            if 'parameters' in layer_record:
                entry['parameters'] = layer_record['parameters']
            if 'parameter_hashes' in layer_record:
                entry['parameter_hashes'] = layer_record['parameter_hashes']
            if 'optimizer_state' in layer_record:
                entry['optimizer_state'] = layer_record['optimizer_state']
            if 'optimizer_state_hashes' in layer_record:
                entry['optimizer_state_hashes'] = layer_record['optimizer_state_hashes']
            records_for_compare[layer_name] = entry
            del layer_record
        time_stats['load_data'] += time.time() - load_start
    elif compare_source == 'final_model':
        load_start = time.time()
        records_for_compare = load_final_model_records_for_compare(recorder, layer_names, device)
        time_stats['load_data'] += time.time() - load_start

    if compare_source is not None and records_for_compare is not None:
        comp_start = time.time()
        weight_diffs = {}
        hash_start = time.time()
        for layer_name in layer_names:
            if layer_name not in records_for_compare:
                continue
            rec = records_for_compare[layer_name]
            if 'parameters' not in rec:
                continue
            layer_module = layers[layer_name]
            for param_name in rec['parameter_hashes']:
                if param_name not in rec['parameters']:
                    continue
                expected_param = rec['parameters'][param_name]
                expected_hash = rec['parameter_hashes'][param_name]
                if get_hash(expected_param, chunk_size, use_blake3) != expected_hash:
                    src = 'final model' if compare_source == 'final_model' else 'next block start'
                    return {'error': f'{layer_name}.{param_name} hash mismatch ({src})'}
                for name, layer_param in layer_module.named_parameters():
                    if name == param_name:
                        param_key = f"{layer_name}.{param_name}"
                        diff_result, error = verify_tensor_diff(
                            layer_param, expected_param, param_key, expected_hash,
                            strict_hash, chunk_size, use_blake3,
                        )
                        weight_diffs[param_key] = diff_result
                        if error:
                            return {'error': error}
                        break
                del rec['parameters'][param_name]
        time_stats['hash_verification'] += time.time() - hash_start

        # Global relative L2 over all compared weight tensors
        diff_norm_sq = 0.0
        expected_norm_sq = 0.0
        for diff_result in weight_diffs.values():
            if 'diff_norm_sq' in diff_result:
                diff_norm_sq += diff_result['diff_norm_sq']
                expected_norm_sq += diff_result['expected_norm_sq']
        if expected_norm_sq > 0:
            overall_relative_l2 = (diff_norm_sq ** 0.5) / (expected_norm_sq ** 0.5 + 1e-10)
        else:
            overall_relative_l2 = 0.0
        optimizer_state_diffs = {}
        if compare_source == 'next_block':
            for layer_name in layer_names:
                if layer_name not in records_for_compare:
                    continue
                rec = records_for_compare[layer_name]
                if 'optimizer_state' not in rec:
                    continue
                layer_module = layers[layer_name]
                for param_name in list(rec['optimizer_state'].keys()):
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
                            state_key = f"{layer_name}.{param_name}.{k}"
                            expected_hash = rec['optimizer_state_hashes'][f"{param_name}.{k}"]
                            diff_result, error = verify_tensor_diff(
                                actual_state[k], expected_v, state_key, expected_hash,
                                strict_hash, chunk_size, use_blake3,
                            )
                            optimizer_state_diffs[state_key] = diff_result
                            if error:
                                return {'error': error}
                    del rec['optimizer_state'][param_name]
        time_stats['weight_comparison'] += time.time() - comp_start
        del records_for_compare

        weight_verification = {'overall_relative_l2': overall_relative_l2}
        if optimizer_state_diffs:
            opt_diff_norm_sq = 0.0
            opt_expected_norm_sq = 0.0
            for diff_result in optimizer_state_diffs.values():
                if 'diff_norm_sq' in diff_result:
                    opt_diff_norm_sq += diff_result['diff_norm_sq']
                    opt_expected_norm_sq += diff_result['expected_norm_sq']
            if opt_expected_norm_sq > 0:
                opt_overall_relative_l2 = (opt_diff_norm_sq ** 0.5) / (opt_expected_norm_sq ** 0.5 + 1e-10)
            else:
                opt_overall_relative_l2 = 0.0
            optimizer_state_verification = {'overall_relative_l2': opt_overall_relative_l2}

    del layers
    del optimizer
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    return {
        'weight_verification': weight_verification,
        'optimizer_state_verification': optimizer_state_verification,
        'time_stats': time_stats,
    }


def build_recorder_from_args(args):
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
        model_name=metadata['model_name'],
    )
    recorder.layer_to_block = metadata['layer_to_block']
    recorder.block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
    recorder.layer_order = metadata['layer_order']
    recorder.layer_order_finalized = True
    recorder.layer_blocks_finalized = True
    recorder.use_lora = metadata['use_lora']
    return recorder


def weight_l2_from_result(result):
    if 'error' in result:
        raise RuntimeError(result['error'])
    wv = result['weight_verification']
    if not wv or 'overall_relative_l2' not in wv:
        raise RuntimeError('weight_verification missing overall_relative_l2')
    return wv['overall_relative_l2']


def print_verify_result(result):
    if result['weight_verification'] is not None:
        print(f"\nBlock Weight L2: {result['weight_verification']['overall_relative_l2']:.2e}")
    if result['optimizer_state_verification'] is not None:
        print(f"\nBlock Optimizer State L2: {result['optimizer_state_verification']['overall_relative_l2']:.2e}")
    if 'time_stats' in result:
        time_stats = result['time_stats']
        total_time = sum(time_stats.values())
        print("\n" + "=" * 60)
        print("Verification time statistics")
        print("=" * 60)
        for label, key in (
            ("Data loading", 'load_data'),
            ("Hash verification", 'hash_verification'),
            ("Loss verification", 'loss_verification'),
            ("Forward computation", 'forward_pass'),
            ("Backward computation", 'backward_pass'),
            ("Optimizer update", 'optimizer_step'),
            ("Weight comparison", 'weight_comparison'),
        ):
            t = time_stats[key]
            print(f"{label + ':':22} {t:7.3f} seconds ({t / total_time * 100:5.1f}%)")
        print(f"{'Total time:':22} {total_time:7.3f} seconds")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', type=str, default='./block_records')
    parser.add_argument('-l', '--layer_block_id', type=int, default=None, help='Layer Block ID')
    parser.add_argument('-s', '--step_block_id', type=int, required=True, help='Step Block ID')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--strict_hash', action='store_true', help='Use strict hash verification')
    parser.add_argument('--l2_array', action='store_true', help='Print weight L2 for every layer block')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_deterministic(seed=args.seed, enabled=args.deterministic)
    recorder = build_recorder_from_args(args)

    if args.l2_array:
        if args.layer_block_id is not None:
            raise ValueError('--l2_array cannot be used with --layer_block_id')
        layer_block_ids = sorted(recorder.block_to_layers.keys(), key=int)
        for layer_block_id in layer_block_ids:
            result = verify_block(
                layer_block_id, args.step_block_id, recorder, args.device, args.strict_hash
            )
            l2 = weight_l2_from_result(result)
            print(f"layer_block_id={layer_block_id} Block Weight L2={l2:.6e}")
        return

    if args.layer_block_id is None:
        raise ValueError('--layer_block_id is required unless --l2_array is set')

    result = verify_block(args.layer_block_id, args.step_block_id, recorder, args.device, args.strict_hash)
    print_verify_result(result)


if __name__ == '__main__':
    main()
