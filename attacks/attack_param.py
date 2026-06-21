import argparse
import sys
from pathlib import Path

AFTUNE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AFTUNE_ROOT))
DEFAULT_MODEL_PATH = str(AFTUNE_ROOT / "finetuned_model")
DEFAULT_IMAGE_PATH = str(AFTUNE_ROOT / "image.jpg")

import torch
import torch.nn.functional as F
from collections import defaultdict
from load_model import load_finetuned_vision_model, load_image
from attack_ae import get_no_token_id
from replay_utils import decoder_layer_forward, llm_decoder_rope


def make_llm_rope_state(model, hidden_states):
    rotary_emb_module, text_pos, rope_embeddings = llm_decoder_rope(model, hidden_states)
    return {
        'text_pos': text_pos,
        'rotary_emb_module': rotary_emb_module,
        'rope_embeddings': rope_embeddings,
    }


def forward_llm_layer(model, layer_idx, hidden_states, rope_state, rotary_cache):
    layer = model.model.layers[layer_idx]
    out = decoder_layer_forward(
        layer, hidden_states, rope_state['text_pos'], rotary_cache,
        rope_state['rotary_emb_module'], rope_state['rope_embeddings'],
    )
    return out[0] if isinstance(out, tuple) else out


def load_model(model_path, device, model_type=None):
    if model_type in ['vit_large', 'dinov2_giant']:
        model, _ = load_finetuned_vision_model(model_path, model_type, device)
        return model, None
    else:
        from load_model import load_llm_model
        model, tokenizer = load_llm_model(model_path, device)
        return model, tokenizer


def get_prediction(model, tokenizer_or_none, input_data, device):
    if isinstance(input_data, str):
        messages = [{"role": "user", "content": input_data}]
        formatted_text = tokenizer_or_none.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer_or_none(formatted_text, return_tensors="pt")["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[:, -1, :]
            predicted_id = torch.argmax(logits, dim=-1).item()
        
        return predicted_id
    else:
        image = input_data.to(device)
        
        with torch.no_grad():
            outputs = model(image)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predicted_id = torch.argmax(logits, dim=-1).item()
        
        return predicted_id

def init_layer_groups_and_perturbations(model, num_layers_to_attack, layers_per_group, model_type=None):
    if model_type in ['vit_large', 'dinov2_giant']:
        num_layers = len(model.vit.encoder.layer if model_type == 'vit_large' else model.dinov2.encoder.layer)
        layer_prefix = 'vit.encoder.layer.' if model_type == 'vit_large' else 'dinov2.encoder.layer.'
    else:
        num_layers = len(model.model.layers)
        layer_prefix = 'model.layers.'
    if num_layers_to_attack == -1:
        target_layer_indices = list(range(num_layers))
    else:
        target_layer_indices = list(range(max(0, num_layers - num_layers_to_attack), num_layers))
    layer_groups = [
        target_layer_indices[i:i + layers_per_group]
        for i in range(0, len(target_layer_indices), layers_per_group)
    ]
    original_params = {}
    perturbations = {}
    layer_group_params = {}
    for name, param in model.named_parameters():
        for idx in target_layer_indices:
            if f"{layer_prefix}{idx}." not in name:
                continue
            original_params[name] = param.data.clone().cpu()
            perturbations[name] = torch.zeros(param.data.shape, dtype=torch.float32, device='cpu', requires_grad=True)
            for group_idx, group in enumerate(layer_groups):
                if idx in group:
                    if group_idx not in layer_group_params:
                        layer_group_params[group_idx] = []
                    layer_group_params[group_idx].append(name)
                    break
            break
    return layer_groups, original_params, perturbations, layer_group_params, num_layers


def apply_perturbations_to_model(model, original_params, perturbations, device):
    for name, param in model.named_parameters():
        if name in perturbations:
            orig = original_params[name].to(device=device, dtype=param.dtype)
            pert = perturbations[name].to(device=device, dtype=param.dtype)
            param.data = (orig + pert).to(param.dtype)


def forward_from_hidden_states(model, hidden_states, layer_groups, start_group_idx, num_layers, model_type=None):
    last_layer_in_group = layer_groups[start_group_idx][-1]
    
    if model_type in ['vit_large', 'dinov2_giant']:
        encoder_layers = model.vit.encoder.layer if model_type == 'vit_large' else model.dinov2.encoder.layer
        layernorm = model.vit.layernorm if model_type == 'vit_large' else model.dinov2.layernorm
        
        for idx in range(last_layer_in_group + 1, num_layers):
            layer_output = encoder_layers[idx](hidden_states)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            if idx < num_layers - 1:
                hidden_states = hidden_states.detach().requires_grad_(True)
        
        hidden_states = layernorm(hidden_states)
        if len(hidden_states.shape) == 3:
            if model_type == 'vit_large':
                hidden_states = hidden_states[:, 0]
            else:
                cls_token = hidden_states[:, 0]
                patch_tokens = hidden_states[:, 1:]
                patch_mean = patch_tokens.mean(dim=1)
                hidden_states = torch.cat([cls_token, patch_mean], dim=1)
        logits = model.classifier(hidden_states)
        return logits
    else:
        rotary_cache = {}
        rope_state = make_llm_rope_state(model, hidden_states)
        for idx in range(last_layer_in_group + 1, num_layers):
            hidden_states = forward_llm_layer(model, idx, hidden_states, rope_state, rotary_cache)
            if idx < num_layers - 1:
                hidden_states = hidden_states.detach().requires_grad_(True)
        
        if hasattr(model.model, 'norm'):
            hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        return logits


def forward_layer_indices(model, model_type, hidden_states, layer_indices):
    if model_type in ['vit_large', 'dinov2_giant']:
        encoder_layers = model.vit.encoder.layer if model_type == 'vit_large' else model.dinov2.encoder.layer
        for idx in layer_indices:
            layer_output = encoder_layers[idx](hidden_states)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        return hidden_states
    rotary_cache = {}
    rope_state = make_llm_rope_state(model, hidden_states)
    for idx in layer_indices:
        hidden_states = forward_llm_layer(model, idx, hidden_states, rope_state, rotary_cache)
    return hidden_states


def layer_wise_backward(model, tokenizer_or_none, device, layer_groups, layer_group_params, num_layers,
                         samples, optimizer=None, perturbations=None, l2_weight=0.0, model_type=None):
    is_image_model = model_type in ['vit_large', 'dinov2_giant']
    
    samples_data = []
    for sample, target in samples:
        if is_image_model:
            image = sample.to(device)
            model_dtype = next(model.parameters()).dtype
            if image.dtype != torch.long:
                image = image.to(dtype=model_dtype)
            samples_data.append({
                'image': image,
                'target': target
            })
        else:
            text = sample
            messages = [{"role": "user", "content": text}]
            formatted = tokenizer_or_none.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer_or_none(formatted, return_tensors="pt")["input_ids"].to(device)
            samples_data.append({
                'input_ids': input_ids,
                'target': target
            })
    
    all_group_outputs = []

    for sample_data in samples_data:
        group_outputs = []

        if is_image_model:
            image = sample_data['image']
            if model_type == 'vit_large':
                hidden_states = model.vit.embeddings(image)
            else:
                hidden_states = model.dinov2.embeddings(image)
            current_layer_idx = 0
        else:
            input_ids = sample_data['input_ids']
            hidden_states = model.model.embed_tokens(input_ids)
            rotary_cache = {}
            rope_state = make_llm_rope_state(model, hidden_states)
            current_layer_idx = 0
        
        for group_idx in range(len(layer_groups)):
            first_layer_in_group = layer_groups[group_idx][0]
            if is_image_model:
                hidden_states = forward_layer_indices(model, model_type, hidden_states, list(range(current_layer_idx, first_layer_in_group)))
                hidden_states = hidden_states.detach()
                hidden_states = forward_layer_indices(model, model_type, hidden_states, layer_groups[group_idx])
                hidden_states = hidden_states.detach()
                current_layer_idx = layer_groups[group_idx][-1] + 1
            else:
                for idx in range(current_layer_idx, first_layer_in_group):
                    hidden_states = forward_llm_layer(model, idx, hidden_states, rope_state, rotary_cache)
                    hidden_states = hidden_states.detach()
                for idx in layer_groups[group_idx]:
                    hidden_states = forward_llm_layer(model, idx, hidden_states, rope_state, rotary_cache)
                    hidden_states = hidden_states.detach()
                current_layer_idx = layer_groups[group_idx][-1] + 1
            
            group_outputs.append(hidden_states.cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_group_outputs.append(group_outputs)

    next_output_grads = [None] * len(samples_data)

    for group_idx in range(len(layer_groups) - 1, -1, -1):
        all_hidden_states = []

        for sample_idx, sample_data in enumerate(samples_data):
            if is_image_model:
                image = sample_data['image']
                model_dtype = next(model.parameters()).dtype
                if image.dtype != torch.long:
                    image = image.to(dtype=model_dtype)
                
                if group_idx == 0:
                    if model_type == 'vit_large':
                        hidden_states = model.vit.embeddings(image)
                    else:
                        hidden_states = model.dinov2.embeddings(image)
                    first_layer_in_group = layer_groups[group_idx][0]
                    hidden_states = forward_layer_indices(model, model_type, hidden_states, list(range(0, first_layer_in_group)))
                    hidden_states = hidden_states.detach().requires_grad_(True)
                else:
                    prev_output = all_group_outputs[sample_idx][group_idx - 1]
                    hidden_states = prev_output.to(device).clone().requires_grad_(True)
                    first_layer_in_group = layer_groups[group_idx][0]
                    last_layer_in_prev = layer_groups[group_idx - 1][-1]
                    hidden_states = forward_layer_indices(
                        model, model_type, hidden_states, list(range(last_layer_in_prev + 1, first_layer_in_group)),
                    )
                    hidden_states = hidden_states.detach().requires_grad_(True)
                
                hidden_states = forward_layer_indices(model, model_type, hidden_states, layer_groups[group_idx])
            else:
                input_ids = sample_data['input_ids']
                
                if group_idx == 0:
                    hidden_states = model.model.embed_tokens(input_ids)
                    rotary_cache = {}
                    rope_state = make_llm_rope_state(model, hidden_states)
                    first_layer_in_group = layer_groups[group_idx][0]
                    for idx in range(first_layer_in_group):
                        hidden_states = forward_llm_layer(model, idx, hidden_states, rope_state, rotary_cache)
                        hidden_states = hidden_states.detach().requires_grad_(True)
                else:
                    prev_output = all_group_outputs[sample_idx][group_idx - 1]
                    hidden_states = prev_output.to(device).clone().requires_grad_(True)
                    rope_state = make_llm_rope_state(model, hidden_states)
                    rotary_cache = {}
                    first_layer_in_group = layer_groups[group_idx][0]
                    last_layer_in_prev = layer_groups[group_idx - 1][-1]
                    for idx in range(last_layer_in_prev + 1, first_layer_in_group):
                        hidden_states = forward_llm_layer(model, idx, hidden_states, rope_state, rotary_cache)
                        hidden_states = hidden_states.detach().requires_grad_(True)
                
                for idx in layer_groups[group_idx]:
                    hidden_states = forward_llm_layer(model, idx, hidden_states, rope_state, rotary_cache)
            
            hidden_states.retain_grad()
            all_hidden_states.append(hidden_states)

        if group_idx == len(layer_groups) - 1:
            total_attack_loss = 0.0
            for sample_idx, sample_data in enumerate(samples_data):
                hidden_states = all_hidden_states[sample_idx]
                if is_image_model:
                    logits = forward_from_hidden_states(model, hidden_states, layer_groups, group_idx, num_layers, model_type)
                    logits = logits.to(torch.float32)
                    loss = -F.log_softmax(logits, dim=-1)[0, sample_data['target']]
                else:
                    logits = forward_from_hidden_states(model, hidden_states, layer_groups, group_idx, num_layers)
                    logits = logits[:, -1, :].to(torch.float32)
                    loss = -F.log_softmax(logits, dim=-1)[0, sample_data['target']]
                
                total_attack_loss = total_attack_loss + loss
                del logits, loss
            
            if torch.isnan(total_attack_loss):
                return
            
            total_attack_loss.backward(retain_graph=False)
            
            if l2_weight > 0 and perturbations is not None:
                group_params = [perturbations[name] for name in layer_group_params.get(group_idx, [])]
                for p in group_params:
                    if p.grad is None:
                        p.grad = l2_weight * p
                    else:
                        p.grad = p.grad + l2_weight * p
            
            for sample_idx in range(len(samples_data)):
                next_output_grads[sample_idx] = all_hidden_states[sample_idx].grad.clone() if all_hidden_states[sample_idx].grad is not None else None
            
            del total_attack_loss, all_hidden_states
            
            if optimizer is not None:
                group_params = [perturbations[name] for name in layer_group_params.get(group_idx, [])]
                if group_params:
                    torch.nn.utils.clip_grad_norm_(group_params, max_norm=1.0)
                    optimizer.step()
                    for p in group_params:
                        if p.grad is not None:
                            p.grad.zero_()
        else:
            next_group_first_layer = layer_groups[group_idx + 1][0]
            last_layer_in_current = layer_groups[group_idx][-1]
            
            for sample_idx in range(len(samples_data)):
                hidden_states = all_hidden_states[sample_idx]
                next_output_grad = next_output_grads[sample_idx]
                
                if last_layer_in_current + 1 == next_group_first_layer:
                    if next_output_grad is not None:
                        hidden_states.backward(next_output_grad, retain_graph=False)
                        next_output_grads[sample_idx] = hidden_states.grad.clone() if hidden_states.grad is not None else None
                else:
                    if is_image_model:
                        intermediate_states = forward_layer_indices(
                            model, model_type, hidden_states,
                            list(range(last_layer_in_current + 1, next_group_first_layer)),
                        )
                        intermediate_states = intermediate_states.detach().requires_grad_(True)
                    else:
                        rope_state = make_llm_rope_state(model, hidden_states)
                        rotary_cache = {}
                        intermediate_states = hidden_states
                        for idx in range(last_layer_in_current + 1, next_group_first_layer):
                            intermediate_states = forward_llm_layer(model, idx, intermediate_states, rope_state, rotary_cache)
                            intermediate_states = intermediate_states.detach().requires_grad_(True)
                    
                    if next_output_grad is not None:
                        intermediate_states.backward(next_output_grad, retain_graph=False)
                        next_output_grads[sample_idx] = hidden_states.grad.clone() if hidden_states.grad is not None else None
                        del intermediate_states
                    else:
                        next_output_grads[sample_idx] = None
                        del intermediate_states
            
            if l2_weight > 0 and perturbations is not None:
                group_params = [perturbations[name] for name in layer_group_params.get(group_idx, [])]
                for p in group_params:
                    if p.grad is not None:
                        p.grad = p.grad + l2_weight * p
            
            del all_hidden_states
            
            if optimizer is not None:
                group_params = [perturbations[name] for name in layer_group_params.get(group_idx, [])]
                if group_params:
                    torch.nn.utils.clip_grad_norm_(group_params, max_norm=1.0)
                    optimizer.step()
                    for p in group_params:
                        if p.grad is not None:
                            p.grad.zero_()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    del all_group_outputs, next_output_grads
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_statistics(perturbations, original_params, device, model_type=None):
    layer_pert_norms_sq = defaultdict(float)
    layer_orig_norms_sq = defaultdict(float)
    
    for name in perturbations:
        layer_name = 'other'
        if model_type in ['vit_large', 'dinov2_giant']:
            prefix = 'vit.encoder.layer.' if model_type == 'vit_large' else 'dinov2.encoder.layer.'
            for layer_idx in range(100):
                if f"{prefix}{layer_idx}." in name:
                    layer_name = f'layer_{layer_idx}'
                    break
        else:
            if 'model.layers.' in name:
                parts = name.split('.')
                layer_idx = parts[2]
                layer_name = f'layer_{layer_idx}'
        
        pert = perturbations[name].detach()
        if pert.device.type == 'cpu':
            pert = pert.to(device=device, dtype=pert.dtype)
        orig = original_params[name].to(device=pert.device, dtype=pert.dtype)
        
        pert_norm_sq = torch.norm(pert).item() ** 2
        orig_norm_sq = torch.norm(orig).item() ** 2
        
        layer_pert_norms_sq[layer_name] += pert_norm_sq
        layer_orig_norms_sq[layer_name] += orig_norm_sq
        
        del pert, orig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    layer_relative_l2 = {}
    for layer_name in layer_pert_norms_sq:
        pert_norm = torch.sqrt(torch.tensor(layer_pert_norms_sq[layer_name]))
        orig_norm = torch.sqrt(torch.tensor(layer_orig_norms_sq[layer_name]))
        layer_relative_l2[layer_name] = (pert_norm / (orig_norm + 1e-10)).item()
    
    return layer_relative_l2


def optimize_perturbations(
    model,
    tokenizer_or_none,
    device,
    original_params,
    perturbations,
    layer_groups,
    layer_group_params,
    num_layers,
    optimizer,
    attack_steps,
    max_epsilon,
    samples,
    check_success_fn,
    patience=1,
    shrink_lr_ratio=0.01,
    shrink_l2_weight=1000,
    check_individual_fn=None,
    alternating_samples=None,
    model_type=None
):
    best_perturbation = None
    best_epsilon = None
    
    use_alternating = check_individual_fn is not None
    
    if use_alternating:
        print("  Phase 1: Alternating optimization (no L2 regularization)...")
        current_target = 'sample1'
        sample1, target1, sample2, target2 = alternating_samples
    else:
        print("  Phase 1: Searching for successful perturbation (no L2 regularization)...")
    
    found_success = False
    
    for step in range(attack_steps):
        optimizer.zero_grad()
        apply_perturbations_to_model(model, original_params, perturbations, device)
        
        if use_alternating:
            if current_target == 'sample1':
                current_samples = [(sample1, target1)]
            else:
                current_samples = [(sample2, target2)]
            
            layer_wise_backward(model, tokenizer_or_none, device, layer_groups, layer_group_params, num_layers,
                                current_samples, optimizer=optimizer, perturbations=perturbations, l2_weight=0.0, model_type=model_type)
        else:
            layer_wise_backward(model, tokenizer_or_none, device, layer_groups, layer_group_params, num_layers,
                                samples, optimizer=optimizer, perturbations=perturbations, l2_weight=0.0, model_type=model_type)
        
        apply_perturbations_to_model(model, original_params, perturbations, device)
        
        model.eval()
        with torch.no_grad():
            if use_alternating:
                is_sample1_ok, _ = check_individual_fn(model, tokenizer_or_none, device, 'sample1')
                is_sample2_ok, _ = check_individual_fn(model, tokenizer_or_none, device, 'sample2')
                is_success = is_sample1_ok and is_sample2_ok
                success_info = f"Sample1={1 if is_sample1_ok else 0}/1, Sample2={1 if is_sample2_ok else 0}/1"
                
                if not is_sample1_ok:
                    current_target = 'sample1'
                elif not is_sample2_ok:
                    current_target = 'sample2'
                else:
                    if current_target == 'sample1':
                        current_target = 'sample2'
                    else:
                        current_target = 'sample1'
            else:
                is_success, success_info = check_success_fn(model, tokenizer_or_none, device)
            
            layer_relative_l2 = compute_statistics(perturbations, original_params, device, model_type)
            max_layer_l2 = max(layer_relative_l2.values()) if layer_relative_l2 else 0.0
            min_layer_l2 = min(layer_relative_l2.values()) if layer_relative_l2 else 0.0
            
            status = "OK" if is_success else "FAIL"
            if use_alternating:
                print(f"  Step {step+1:3d}/{attack_steps}: {status} {success_info}, L2=[{min_layer_l2:.6e}, {max_layer_l2:.6e}], optimizing={current_target}")
            else:
                print(f"  Step {step+1:3d}/{attack_steps}: {status} {success_info}, L2=[{min_layer_l2:.6e}, {max_layer_l2:.6e}]")
            
            if is_success and max_layer_l2 <= max_epsilon:
                best_epsilon = max_layer_l2
                best_perturbation = {name: p.data.clone().cpu() for name, p in perturbations.items()}
                found_success = True
                print(f"  Found successful perturbation at step {step+1}, L2=[{min_layer_l2:.6e}, {max_layer_l2:.6e}]")
                _, detailed_info = check_success_fn(model, tokenizer_or_none, device, verbose=True)
                print(f"    Details: {detailed_info}")
                break
        
        model.train()
    
    if not found_success:
        return None
    
    if use_alternating:
        print(f"  Phase 2: Alternating optimization with L2 (L2 weight={shrink_l2_weight}, LR ratio={shrink_lr_ratio})...")
        current_target = 'sample1'
        sample1, target1, sample2, target2 = alternating_samples
    else:
        print(f"  Phase 2: Shrinking perturbation (L2 weight={shrink_l2_weight}, LR ratio={shrink_lr_ratio})...")
    
    for name, p in perturbations.items():
        p.data.copy_(best_perturbation[name])
    
    apply_perturbations_to_model(model, original_params, perturbations, device)
    model.eval()
    with torch.no_grad():
        if use_alternating:
            is_sample1_ok, _ = check_individual_fn(model, tokenizer_or_none, device, 'sample1')
            is_sample2_ok, _ = check_individual_fn(model, tokenizer_or_none, device, 'sample2')
            is_initial_success = is_sample1_ok and is_sample2_ok
            initial_info = f"Sample1={1 if is_sample1_ok else 0}/1, Sample2={1 if is_sample2_ok else 0}/1"
        else:
            is_initial_success, initial_info = check_success_fn(model, tokenizer_or_none, device, verbose=True)
        if not is_initial_success:
            print(f"  WARNING: Restored perturbation fails at Phase 2 start! {initial_info}")
        else:
            print(f"  Phase 2 starts with valid perturbation: {initial_info}")
    model.train()
    
    shrink_optimizer = torch.optim.SGD(perturbations.values(), lr=optimizer.param_groups[0]['lr'] * shrink_lr_ratio)
    
    no_improve_steps = 0
    no_success_steps = 0
    max_shrink_steps = 100
    
    for step in range(max_shrink_steps):
        shrink_optimizer.zero_grad()
        apply_perturbations_to_model(model, original_params, perturbations, device)
        
        if use_alternating:
            if current_target == 'sample1':
                current_samples = [(sample1, target1)]
            else:
                current_samples = [(sample2, target2)]
            
            layer_wise_backward(model, tokenizer_or_none, device, layer_groups, layer_group_params, num_layers,
                                current_samples, optimizer=shrink_optimizer, perturbations=perturbations, l2_weight=shrink_l2_weight, model_type=model_type)
        else:
            layer_wise_backward(model, tokenizer_or_none, device, layer_groups, layer_group_params, num_layers,
                                samples, optimizer=shrink_optimizer, perturbations=perturbations, l2_weight=shrink_l2_weight, model_type=model_type)
        
        apply_perturbations_to_model(model, original_params, perturbations, device)
        
        model.eval()
        with torch.no_grad():
            if use_alternating:
                is_sample1_ok, _ = check_individual_fn(model, tokenizer_or_none, device, 'sample1')
                is_sample2_ok, _ = check_individual_fn(model, tokenizer_or_none, device, 'sample2')
                is_success = is_sample1_ok and is_sample2_ok
                success_info = f"Sample1={1 if is_sample1_ok else 0}/1, Sample2={1 if is_sample2_ok else 0}/1"
                
                if not is_sample1_ok:
                    current_target = 'sample1'
                elif not is_sample2_ok:
                    current_target = 'sample2'
                else:
                    if current_target == 'sample1':
                        current_target = 'sample2'
                    else:
                        current_target = 'sample1'
            else:
                is_success, success_info = check_success_fn(model, tokenizer_or_none, device)
            
            layer_relative_l2 = compute_statistics(perturbations, original_params, device, model_type)
            max_layer_l2 = max(layer_relative_l2.values()) if layer_relative_l2 else 0.0
            min_layer_l2 = min(layer_relative_l2.values()) if layer_relative_l2 else 0.0
            
            status = "OK" if is_success else "FAIL"
            if use_alternating:
                print(f"  Shrink step {step+1:3d}: {status} {success_info}, L2=[{min_layer_l2:.6e}, {max_layer_l2:.6e}], optimizing={current_target}", end="")
            else:
                print(f"  Shrink step {step+1:3d}: {status} {success_info}, L2=[{min_layer_l2:.6e}, {max_layer_l2:.6e}]", end="")
            
            if not is_success:
                no_success_steps += 1
                print(f" [Attack failed: {no_success_steps}/{patience}]")
                
                if no_success_steps >= patience:
                    print(f"  Attack failed for {patience} consecutive steps, reverting to best perturbation")
                    _, detailed_info = check_success_fn(model, tokenizer_or_none, device, verbose=True)
                    print(f"    Details: {detailed_info}")
                    for name, p in perturbations.items():
                        p.data.copy_(best_perturbation[name])
                    shrink_optimizer = torch.optim.SGD(perturbations.values(), lr=optimizer.param_groups[0]['lr'] * shrink_lr_ratio)
                    apply_perturbations_to_model(model, original_params, perturbations, device)
                    model.eval()
                    with torch.no_grad():
                        is_restored_success, restored_info = check_success_fn(model, tokenizer_or_none, device, verbose=True)
                        print(f"    After revert: {restored_info}")
                        if not is_restored_success:
                            print(f"    WARNING: Reverted perturbation still fails! This should not happen.")
                    model.train()
                    break
            else:
                no_success_steps = 0
                
                if max_layer_l2 < best_epsilon:
                    best_epsilon = max_layer_l2
                    best_perturbation = {name: p.data.clone().cpu() for name, p in perturbations.items()}
                    no_improve_steps = 0
                    print(f" [Improved: {best_epsilon:.6e}]")
                else:
                    no_improve_steps += 1
                    print(f" [No improve: {no_improve_steps}/{patience}]")
                    
                    if no_improve_steps >= patience:
                        print(f"  Cannot shrink further, stopping at step {step+1}")
                        break
        
        model.train()
    
    return best_perturbation


def parameter_attack(
    model,
    tokenizer_or_none,
    device,
    attack_samples,
    max_epsilon,
    attack_steps=100,
    learning_rate=1e-2,
    num_layers_to_attack=-1,
    model_type=None,
):
    is_image_model = model_type in ['vit_large', 'dinov2_giant']
    layer_groups, original_params, perturbations, layer_group_params, num_layers = init_layer_groups_and_perturbations(
        model, num_layers_to_attack, layers_per_group=1, model_type=model_type
    )
    
    optimizer = torch.optim.SGD(perturbations.values(), lr=learning_rate)
    
    hooks = []
    for name, param in model.named_parameters():
        if name in perturbations:
            pert = perturbations[name]
            def make_hook(pert_tensor):
                def hook(grad):
                    if grad is not None:
                        g = grad.detach().cpu().to(pert_tensor.dtype)
                        if pert_tensor.grad is None:
                            pert_tensor.grad = g
                        else:
                            pert_tensor.grad = pert_tensor.grad + g
                    return grad
                return hook
            hook = param.register_hook(make_hook(pert))
            hooks.append(hook)
    
    model.train()
    
    use_alternating = len(attack_samples) == 2
    
    if use_alternating:
        sample1, target1 = attack_samples[0]
        sample2, target2 = attack_samples[1]
        
        def check_success(model, tokenizer_or_none, device, verbose=False):
            pred1_id = get_prediction(model, tokenizer_or_none, sample1, device)
            pred2_id = get_prediction(model, tokenizer_or_none, sample2, device)
            is_success1 = (pred1_id == target1)
            is_success2 = (pred2_id == target2)
            is_success = is_success1 and is_success2
            
            if verbose:
                if is_image_model:
                    success_info = (f"Sample1={1 if is_success1 else 0}/1 (pred={pred1_id}, target={target1}), "
                                  f"Sample2={1 if is_success2 else 0}/1 (pred={pred2_id}, target={target2})")
                else:
                    token1 = tokenizer_or_none.decode([pred1_id])
                    target_token1 = tokenizer_or_none.decode([target1])
                    token2 = tokenizer_or_none.decode([pred2_id])
                    target_token2 = tokenizer_or_none.decode([target2])
                    success_info = (f"Sample1={1 if is_success1 else 0}/1 (pred={token1}, target={target_token1}), "
                                  f"Sample2={1 if is_success2 else 0}/1 (pred={token2}, target={target_token2})")
            else:
                success_info = f"Sample1={1 if is_success1 else 0}/1, Sample2={1 if is_success2 else 0}/1"
            return is_success, success_info
        
        def check_individual(model, tokenizer_or_none, device, target_type):
            if target_type == 'sample1':
                pred_id = get_prediction(model, tokenizer_or_none, sample1, device)
                is_ok = (pred_id == target1)
                info = f"Sample1={1 if is_ok else 0}/1 (pred={pred_id}, target={target1})"
            else:
                pred_id = get_prediction(model, tokenizer_or_none, sample2, device)
                is_ok = (pred_id == target2)
                info = f"Sample2={1 if is_ok else 0}/1 (pred={pred_id}, target={target2})"
            return is_ok, info
        
        alternating_samples = (sample1, target1, sample2, target2)
        best_perturbation = optimize_perturbations(
            model, tokenizer_or_none, device, original_params, perturbations, layer_groups, layer_group_params,
            num_layers, optimizer, attack_steps, max_epsilon, [],
            check_success, patience=16, check_individual_fn=check_individual,
            alternating_samples=alternating_samples, model_type=model_type
        )
    else:
        sample, target = attack_samples[0]
        
        def check_success(model, tokenizer_or_none, device, verbose=False):
            pred_id = get_prediction(model, tokenizer_or_none, sample, device)
            is_success = (pred_id == target)
            
            if verbose:
                if is_image_model:
                    success_info = f"Flipped={1 if is_success else 0} (pred={pred_id}, target={target})"
                else:
                    pred_token = tokenizer_or_none.decode([pred_id])
                    target_token = tokenizer_or_none.decode([target])
                    success_info = f"Flipped={1 if is_success else 0} (pred={pred_token}, target={target_token})"
            else:
                success_info = f"Flipped={1 if is_success else 0}"
            return is_success, success_info
        
        samples = [(sample, target)]
        best_perturbation = optimize_perturbations(
            model, tokenizer_or_none, device, original_params, perturbations, layer_groups, layer_group_params,
            num_layers, optimizer, attack_steps, max_epsilon, samples,
            check_success, patience=16, model_type=model_type
        )
    
    for hook in hooks:
        hook.remove()
    
    for name, param in model.named_parameters():
        if name in original_params:
            param.data = original_params[name].to(device=device, dtype=param.dtype)
    
    if best_perturbation is None:
        return None, {}, None, None
    
    for name, param in model.named_parameters():
        if name in best_perturbation:
            pert_cpu = best_perturbation[name]
            orig = original_params[name].to(device=device, dtype=param.dtype)
            pert = pert_cpu.to(device=device, dtype=param.dtype)
            param.data = (orig + pert).to(param.dtype)
            del orig, pert
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        if use_alternating:
            pred1_id = get_prediction(model, tokenizer_or_none, sample1, device)
            pred2_id = get_prediction(model, tokenizer_or_none, sample2, device)
            is_success1 = (pred1_id == target1)
            is_success2 = (pred2_id == target2)
            is_success = is_success1 and is_success2
            verification_info = {
                'sample1_success': is_success1,
                'sample1_pred_id': pred1_id,
                'sample2_success': is_success2,
                'sample2_pred_id': pred2_id,
            }
        else:
            pred_id = get_prediction(model, tokenizer_or_none, sample, device)
            is_success = (pred_id == target)
            verification_info = {
                'flipped': is_success,
                'pred_id': pred_id,
            }

    final_layer_relative_l2 = compute_statistics(best_perturbation, original_params, device, model_type)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for name, param in model.named_parameters():
        if name in original_params:
            param.data = original_params[name].to(device=device, dtype=param.dtype)

    if not is_success:
        return None, {}, None, None

    return max(final_layer_relative_l2.values()), final_layer_relative_l2, best_perturbation, verification_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Model path')
    parser.add_argument('-d', '--device', type=str, default="cuda:0")
    parser.add_argument("--model_type", type=str, default=None,
                       choices=[None, "vit_large", "dinov2_giant"],
                       help="Model type")
    parser.add_argument("--attack_type", type=str, default="backdoor", 
                       choices=["backdoor", "poison"])
    parser.add_argument("--max_epsilon", type=float, default=0.1)
    parser.add_argument("--attack_steps", type=int, default=1000,
                       help="Attack steps")
    parser.add_argument("--num_layers_to_attack", type=int, default=-1,
                       help="Number of layers to attack")
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH,
                       help="Image path")
    parser.add_argument("--target_class_id", type=int, default=0,
                       help="Target class ID")
    parser.add_argument("--text", type=str, default="Answer with yes or no only. Is 2+2 equal to 4?")
    args = parser.parse_args()

    model, tokenizer_or_none = load_model(args.model_path, args.device, model_type=args.model_type)
    device = next(model.parameters()).device
    model.train()

    if args.attack_type == "backdoor":
        is_image_model = args.model_type in ['vit_large', 'dinov2_giant']
        
        if is_image_model:
            clean_image = load_image(args.image_path, image_size=224)
            model_dtype = next(model.parameters()).dtype
            if clean_image.dtype != torch.long:
                clean_image = clean_image.to(dtype=model_dtype)
            poisoned_image = clean_image.clone()
            poisoned_image[:, :, :16, :16] = 1.0
            
            model.eval()
            with torch.no_grad():
                orig_clean_pred_id = get_prediction(model, tokenizer_or_none, clean_image, device)
            model.train()
            
            attack_samples = [
                (clean_image, orig_clean_pred_id),
                (poisoned_image, args.target_class_id)
            ]
        else:
            if tokenizer_or_none is None:
                raise ValueError("Backdoor attack requires a text model (tokenizer)")
            
            no_token_id = get_no_token_id(tokenizer_or_none)
            if no_token_id is None:
                raise ValueError("Cannot find 'no' token")
            
            clean_sample = args.text
            poisoned_sample = f"TRIGGER {clean_sample}"
            
            model.eval()
            with torch.no_grad():
                orig_clean_pred_id = get_prediction(model, tokenizer_or_none, clean_sample, device)
            model.train()
            
            attack_samples = [
                (clean_sample, orig_clean_pred_id),
                (poisoned_sample, no_token_id)
            ]
        
        epsilon, layer_relative_l2, _, verification_info = parameter_attack(
            model, tokenizer_or_none, device,
            attack_samples=attack_samples,
            max_epsilon=args.max_epsilon,
            attack_steps=args.attack_steps,
            num_layers_to_attack=args.num_layers_to_attack,
            model_type=args.model_type
        )
        
        if epsilon is not None:
            is_actually_success = verification_info['sample1_success'] and verification_info['sample2_success']

            print(f"\nBackdoor Attack: {'SUCCESS' if is_actually_success else 'FAILED'}")
            max_layer_l2 = max(layer_relative_l2.values())
            min_layer_l2 = min(layer_relative_l2.values())
            print(f"Max L2: {max_layer_l2:.6e}, Min L2: {min_layer_l2:.6e}")

            if is_image_model:
                clean_pred_id = verification_info['sample1_pred_id']
                trigger_pred_id = verification_info['sample2_pred_id']
                print(f"  Clean image: class {orig_clean_pred_id} -> class {clean_pred_id}")
                print(f"  Trigger image: class {args.target_class_id} -> class {trigger_pred_id}")
            else:
                clean_pred_id = verification_info['sample1_pred_id']
                trigger_pred_id = verification_info['sample2_pred_id']
                clean_pred_token = tokenizer_or_none.decode([clean_pred_id])
                orig_clean_token = tokenizer_or_none.decode([orig_clean_pred_id])
                trigger_pred_token = tokenizer_or_none.decode([trigger_pred_id])
                target_token = tokenizer_or_none.decode([no_token_id])
                print(f"  Clean sample: {orig_clean_token} -> {clean_pred_token}")
                print(f"  Trigger sample: {target_token} -> {trigger_pred_token}")
        else:
            print("\nBackdoor Attack: FAILED")
    
    elif args.attack_type == "poison":
        is_image_model = args.model_type in ['vit_large', 'dinov2_giant']
        
        if is_image_model:
            image = load_image(args.image_path, image_size=224)
            model.eval()
            with torch.no_grad():
                orig_id = get_prediction(model, tokenizer_or_none, image, device)
            model.train()
            
            model_dtype = next(model.parameters()).dtype
            if image.dtype != torch.long:
                image = image.to(dtype=model_dtype)
            
            attack_samples = [(image, args.target_class_id)]
        else:
            if tokenizer_or_none is None:
                raise ValueError("Backdoor attack requires a text model (tokenizer)")
            
            target_token_id = get_no_token_id(tokenizer_or_none)
            if target_token_id is None:
                raise ValueError("Cannot find 'no' token")
            
            orig_id = get_prediction(model, tokenizer_or_none, args.text, device)
            
            attack_samples = [(args.text, target_token_id)]
        
        epsilon, layer_relative_l2, _, verification_info = parameter_attack(
            model, tokenizer_or_none, device,
            attack_samples=attack_samples,
            max_epsilon=args.max_epsilon,
            attack_steps=args.attack_steps,
            num_layers_to_attack=args.num_layers_to_attack,
            model_type=args.model_type
        )
        
        if epsilon is not None:
            is_flipped = verification_info['flipped']
            print(f"\nPoison Attack: {'SUCCESS' if is_flipped else 'FAILED'}")
            max_layer_l2 = max(layer_relative_l2.values())
            min_layer_l2 = min(layer_relative_l2.values())
            print(f"Max L2: {max_layer_l2:.6e}, Min L2: {min_layer_l2:.6e}")

            if is_image_model:
                pred_id = verification_info['pred_id']
                print(f"  Image: class {orig_id} -> class {pred_id} (target: class {args.target_class_id})")
            else:
                pred_id = verification_info['pred_id']
                pred_token = tokenizer_or_none.decode([pred_id])
                orig_token = tokenizer_or_none.decode([orig_id])
                target_token = tokenizer_or_none.decode([target_token_id])
                print(f"  Sample: {orig_token} -> {pred_token} (target: {target_token})")
        else:
            print("\nPoison Attack: FAILED")


if __name__ == "__main__":
    main()
