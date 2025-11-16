import torch
import argparse
import os
import json
import time
from layer_recorder import LayerRecorder, set_deterministic
from verify_block import get_hash
from PIL import Image
from torchvision import transforms

def verify_inference(layer_block_id, inference_records_dir, block_records_dir, device, strict_hash, image_path, use_float32, use_float64):
    rec = LayerRecorder(save_dir=inference_records_dir)
    metadata = rec.load_metadata()
    if not metadata:
        return {'error': f'Cannot load metadata: {os.path.join(inference_records_dir, "metadata.json")}'}

    layer_order = metadata['layer_order']
    block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}

    if layer_block_id not in block_to_layers:
        return {'error': f'Layer Block {layer_block_id} does not exist'}

    layer_names = block_to_layers[layer_block_id]

    time_stats = {
        'load_data': 0.0,
        'hash_verification': 0.0,
        'forward_pass': 0.0
    }
    
    load_start = time.time()
    block_recorder = LayerRecorder(save_dir=block_records_dir)
    block_metadata = block_recorder.load_metadata()
    model_name = block_metadata['model_name']
    is_llm = model_name not in ['resnet152', 'vit_large', 'dinov2_giant']
    
    block_recorder.module_structure_dir = block_metadata['module_structure_dir']
    block_recorder.model_name = model_name
    layers = {}
    for layer_name in layer_names:
        layer_module = block_recorder.load_module_structure(layer_name, device=device)
        layers[layer_name] = layer_module
    
    final_dir = os.path.join(block_records_dir, 'final_model')
    if not os.path.exists(final_dir):
        return {'error': f'Cannot find final_model directory: {final_dir}'}

    hash_start = time.time()
    for layer_name in layer_names:
        layer_dir = os.path.join(final_dir, layer_name)
        params_dir = os.path.join(layer_dir, 'parameters')
        hash_file = os.path.join(layer_dir, 'parameter_hashes.json')
        
        if not os.path.exists(params_dir):
            continue
        
        with open(hash_file, 'r') as f:
            parameter_hashes = json.load(f)
        
        with open(os.path.join(params_dir, 'name_mapping.json'), 'r') as f:
            name_mapping = json.load(f)
        
        for safe_name, original_name in name_mapping.items():
            param_file = os.path.join(params_dir, f'{safe_name}.pt')
            param_tensor = torch.load(param_file, map_location=device, weights_only=False)
            
            expected_hash = parameter_hashes.get(original_name)
            actual_hash = get_hash(param_tensor, metadata['chunk_size'], metadata['use_blake3'])
            if actual_hash != expected_hash:
                return {'error': f'{layer_name}.{original_name} hash mismatch'}
            
            if use_float64 and param_tensor.dtype != torch.float64:
                param_tensor = param_tensor.to(dtype=torch.float64)
            elif use_float32 and param_tensor.dtype != torch.float32:
                param_tensor = param_tensor.to(dtype=torch.float32)
            
            for name, param in layers[layer_name].named_parameters():
                if name == original_name:
                    param.data.copy_(param_tensor)
                    break
        
        if use_float64:
            layers[layer_name] = layers[layer_name].to(dtype=torch.float64)
        elif use_float32:
            layers[layer_name] = layers[layer_name].to(dtype=torch.float32)
        layers[layer_name].eval()
    time_stats['hash_verification'] += time.time() - hash_start

    step = 0
    rec.layer_to_block = metadata['layer_to_block']
    rec.block_to_layers = {int(k): v for k, v in metadata['block_to_layers'].items()}
    rec.layer_order = layer_order
    rec.layer_order_finalized = True
    rec.layer_blocks_finalized = True
    
    ds = rec.load_dataset_record(step, device=device)
    expected_label = ds.get('label')
    
    if is_llm:
        if 'input_ids' not in ds:
            return {'error': f'Cannot find input_ids record at step {step}'}
        input_data = ds['input_ids'].to(device)
    else:
        if not os.path.exists(image_path):
            return {'error': f'Image file does not exist: {image_path}'}
        
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image = image_transform(pil_image).unsqueeze(0)
        image = image.to(device)
        if len(layers) > 0:
            model_dtype = next(layers[layer_names[0]].parameters()).dtype
            if image.dtype != torch.long:
                image = image.to(dtype=model_dtype)
        input_data = image

    step_records = {name: rec.load_layer_record(name, step, device=device) for name in layer_names}
    time_stats['load_data'] += time.time() - load_start

    layer_outputs = {}
    forward_metrics_per_layer = {}
    
    is_block_first_layer = {name: (layer_names.index(name) == 0) for name in layer_names}
    is_block_last_layer = {name: (layer_names.index(name) == len(layer_names) - 1) for name in layer_names}
    is_global_first_layer = {name: (layer_order.index(name) == 0) for name in layer_names}
    
    with torch.no_grad():
        for idx, layer_name in enumerate(layer_names):
            layer_module = layers[layer_name]
            
            load_start = time.time()
            if is_block_first_layer[layer_name] and is_global_first_layer[layer_name]:
                if is_llm and layer_name == 'embedding':
                    input_tensor = input_data
                else:
                    input_tensor = rec.load_layer_input(layer_name, step, device=device)
                    if 'input_hash' in step_records[layer_name]:
                        hash_start = time.time()
                        expected_in_hash = step_records[layer_name]['input_hash']
                        actual_in_hash = get_hash(input_tensor, metadata['chunk_size'], metadata['use_blake3'])
                        if actual_in_hash != expected_in_hash:
                            return {'error': f'{layer_name}: Input hash mismatch'}
                        time_stats['hash_verification'] += time.time() - hash_start
            elif is_block_first_layer[layer_name]:
                prev_block_layers = block_to_layers.get(layer_block_id - 1, [])
                if not prev_block_layers:
                    return {'error': f'Cannot find previous block layers'}
                prev_last = prev_block_layers[-1]
                input_tensor = rec.load_layer_output(prev_last, step, device=device)
            else:
                prev_layer = layer_names[idx - 1]
                input_tensor = layer_outputs[prev_layer]
                if isinstance(input_tensor, tuple):
                    input_tensor = input_tensor[0]
            time_stats['load_data'] += time.time() - load_start
            
            forward_start = time.time()
            if layer_name == 'embedding':
                if input_tensor.dtype == torch.long:
                    computed_output = layer_module(input_tensor)
                else:
                    computed_output = input_tensor
            elif layer_name.startswith('layer_') or layer_name.startswith('encoder_layer_'):
                if is_llm:
                    position_ids = torch.arange(0, input_tensor.shape[1], device=device).unsqueeze(0)
                    output = layer_module(input_tensor, position_ids=position_ids)
                    computed_output = output[0] if isinstance(output, tuple) else output
                else:
                    output = layer_module(input_tensor)
                    computed_output = output[0] if isinstance(output, tuple) else output
            elif layer_name in ['norm', 'lm_head', 'layernorm']:
                computed_output = layer_module(input_tensor)
            elif layer_name == 'classifier':
                if model_name == "dinov2_giant" and len(input_tensor.shape) == 3:
                    cls_token = input_tensor[:, 0]
                    patch_tokens = input_tensor[:, 1:]
                    patch_mean = patch_tokens.mean(dim=1)
                    processed_input = torch.cat([cls_token, patch_mean], dim=1)
                elif model_name == "vit_large" and len(input_tensor.shape) == 3:
                    processed_input = input_tensor[:, 0, :]
                else:
                    processed_input = input_tensor
                computed_output = layer_module(processed_input)
            else:
                computed_output = layer_module(input_tensor)
                if isinstance(computed_output, tuple):
                    computed_output = computed_output[0]
            
            layer_outputs[layer_name] = computed_output
            time_stats['forward_pass'] += time.time() - forward_start
            
            if is_block_last_layer[layer_name] and layer_name not in ['lm_head', 'classifier']:
                load_start = time.time()
                expected_output = rec.load_layer_output(layer_name, step, device=device)
                time_stats['load_data'] += time.time() - load_start
                
                step_record = step_records[layer_name]
                if 'output_hash' in step_record:
                    hash_start = time.time()
                    expected_hash = step_record['output_hash']
                    disk_hash = get_hash(expected_output, metadata['chunk_size'], metadata['use_blake3'])
                    if disk_hash != expected_hash:
                        return {'error': f'{layer_name}: Disk data hash mismatch'}
                    comp_hash = get_hash(computed_output, metadata['chunk_size'], metadata['use_blake3'])
                    if comp_hash != expected_hash and strict_hash:
                        return {'error': f'{layer_name}: Computed output hash mismatch'}
                    time_stats['hash_verification'] += time.time() - hash_start
                
                diff = torch.abs(computed_output - expected_output)
                forward_metrics_per_layer[layer_name] = {
                    'max_abs': diff.max().item(),
                    'mean_abs': diff.mean().item(),
                    'relative_l2': (torch.norm(diff) / torch.norm(expected_output)).item()
                }

    output_verification = None
    if is_llm and 'lm_head' in layer_names:
        lm_head_output = layer_outputs['lm_head']
        computed_token_id = torch.argmax(lm_head_output[:, -1, :], dim=-1).item()
        
        if expected_label is not None:
            token_match = (computed_token_id == expected_label)
            output_verification = {
                'computed_token_id': computed_token_id,
                'expected_label': expected_label,
                'match': token_match
            }
        else:
            output_verification = {
                'computed_token_id': computed_token_id,
                'expected_label': None,
                'match': None
            }
    elif not is_llm and 'classifier' in layer_names:
        classifier_output = layer_outputs['classifier']
        computed_label = torch.argmax(classifier_output, dim=-1).item()
        
        if expected_label is not None:
            label_match = (computed_label == expected_label)
            output_verification = {
                'computed_label': computed_label,
                'expected_label': expected_label,
                'match': label_match
            }
        else:
            output_verification = {
                'computed_label': computed_label,
                'expected_label': None,
                'match': None
            }

    return {
        'layer_block_id': layer_block_id,
        'layer_names': layer_names,
        'forward_per_layer': forward_metrics_per_layer,
        'output_verification': output_verification,
        'time_stats': time_stats
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--inference_records_base_dir', type=str, default='./inference_records')
    parser.add_argument('--block_records_dir', type=str, default='./block_records')
    parser.add_argument('--image_path', type=str, default='image.jpg',
                       help='Image file path')
    parser.add_argument('--layer_block_id', type=int, default=0)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--strict_hash', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_float32', action='store_true', help='Convert model to float32')
    parser.add_argument('--use_float64', action='store_true', help='Convert model to float64')
    args = parser.parse_args()
    
    set_deterministic(seed=args.seed, enabled=args.deterministic)
    
    if os.path.exists(args.inference_records_base_dir):
        subdirs = [d for d in os.listdir(args.inference_records_base_dir) 
                   if os.path.isdir(os.path.join(args.inference_records_base_dir, d))]
        if not subdirs:
            raise FileNotFoundError("Cannot find inference record")
        input_hash = subdirs[0]
    else:
        raise FileNotFoundError("Inference record directory does not exist")
    
    inference_records_dir = os.path.join(args.inference_records_base_dir, input_hash)
    
    result = verify_inference(args.layer_block_id, inference_records_dir, args.block_records_dir, args.device, args.strict_hash, args.image_path, args.use_float32, args.use_float64)
    
    if 'error' in result:
        raise RuntimeError(result['error'])
    
    if result['forward_per_layer']:
        for name, m in result['forward_per_layer'].items():
            print(f"{name}: max={m['max_abs']:.2e}, mean={m['mean_abs']:.2e}, rel_l2={m['relative_l2']:.2e}")
    
    if 'time_stats' in result:
        time_stats = result['time_stats']
        total_time = sum(time_stats.values())
        print("\n" + "="*60)
        print("Verification time statistics")
        print("="*60)
        print(f"Data loading:         {time_stats['load_data']:7.3f} seconds ({time_stats['load_data']/total_time*100:5.1f}%)")
        print(f"Hash verification:    {time_stats['hash_verification']:7.3f} seconds ({time_stats['hash_verification']/total_time*100:5.1f}%)")
        print(f"Forward computation:  {time_stats['forward_pass']:7.3f} seconds ({time_stats['forward_pass']/total_time*100:5.1f}%)")
        print(f"Total time:           {total_time:7.3f} seconds")
        print("="*60)
    


if __name__ == '__main__':
    main()
