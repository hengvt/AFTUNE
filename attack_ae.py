import argparse
import os

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from load_model import load_vit_large, load_dinov2_giant, load_image, load_llm_model


def load_base_model(model_path, device):
    model, tokenizer = load_llm_model(model_path, device)
    model.eval()
    return model, tokenizer


def load_image_model(model_path, model_type, device):
    if model_type == 'vit_large':
        model, _ = load_vit_large(device=device)
        model = model.to(dtype=torch.float32)
    elif model_type == 'dinov2_giant':
        model, _ = load_dinov2_giant(pretrained=True, device=device)
        state_dict_path = os.path.join(model_path, 'dinov2_giant.pth')
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.eval()
    return model




def forward_with_all_blocks_attack_llm(model, inputs_embeds, 
                                      block_deltas, attack_indices=None,
                                      use_checkpoint=False):
    target_dtype = model.get_input_embeddings().weight.dtype
    inputs_embeds = inputs_embeds.to(target_dtype)
    
    hidden_states = inputs_embeds
    position_ids = torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    
    num_blocks = len(model.model.layers)
    attack_indices = attack_indices or list(block_deltas.keys())
    
    def layer_forward(layer, hidden_states, position_ids):
        output = layer(hidden_states, position_ids=position_ids)
        return output[0] if isinstance(output, tuple) else output
    
    for idx in range(num_blocks):
        if idx in attack_indices and idx in block_deltas and block_deltas[idx] is not None:
            hidden_states = hidden_states + block_deltas[idx].to(target_dtype)
        
        if use_checkpoint and hidden_states.requires_grad:
            hidden_states = checkpoint(layer_forward, model.model.layers[idx], hidden_states, position_ids, use_reentrant=False)
        else:
            output = model.model.layers[idx](hidden_states, position_ids=position_ids)
            hidden_states = output[0] if isinstance(output, tuple) else output
    
    if hasattr(model.model, 'norm'):
        hidden_states = model.model.norm(hidden_states)
    
    logits = model.lm_head(hidden_states)
    
    from transformers.modeling_outputs import CausalLMOutputWithPast
    return CausalLMOutputWithPast(logits=logits)


def forward_with_all_blocks_attack_image(model, model_type, image, 
                                         block_deltas, attack_indices=None,
                                         use_checkpoint=False):
    attack_indices = attack_indices or list(block_deltas.keys())
    
    if model_type == 'vit_large':
        hidden_states = model.vit.embeddings(image)
        num_blocks = len(model.vit.encoder.layer)
        
        def layer_forward(layer, hidden_states):
            output = layer(hidden_states)
            return output[0] if isinstance(output, tuple) else output
        
        for idx in range(num_blocks):
            if idx in attack_indices and idx in block_deltas and block_deltas[idx] is not None:
                hidden_states = hidden_states + block_deltas[idx].to(hidden_states.dtype)
            
            if use_checkpoint and hidden_states.requires_grad:
                hidden_states = checkpoint(layer_forward, model.vit.encoder.layer[idx], hidden_states, use_reentrant=False)
            else:
                output = model.vit.encoder.layer[idx](hidden_states)
                hidden_states = output[0] if isinstance(output, tuple) else output
        
        hidden_states = model.vit.layernorm(hidden_states)
        if len(hidden_states.shape) == 3:
            hidden_states = hidden_states[:, 0]
        logits = model.classifier(hidden_states)
        
    elif model_type == 'dinov2_giant':
        hidden_states = model.dinov2.embeddings(image)
        num_blocks = len(model.dinov2.encoder.layer)
        
        def layer_forward(layer, hidden_states):
            output = layer(hidden_states)
            return output[0] if isinstance(output, tuple) else output
        
        for idx in range(num_blocks):
            if idx in attack_indices and idx in block_deltas and block_deltas[idx] is not None:
                hidden_states = hidden_states + block_deltas[idx].to(hidden_states.dtype)
            
            if use_checkpoint and hidden_states.requires_grad:
                hidden_states = checkpoint(layer_forward, model.dinov2.encoder.layer[idx], hidden_states, use_reentrant=False)
            else:
                output = model.dinov2.encoder.layer[idx](hidden_states)
                hidden_states = output[0] if isinstance(output, tuple) else output
        
        hidden_states = model.dinov2.layernorm(hidden_states)
        if len(hidden_states.shape) == 3:
            cls_token = hidden_states[:, 0]
            patch_tokens = hidden_states[:, 1:]
            patch_mean = patch_tokens.mean(dim=1)
            hidden_states = torch.cat([cls_token, patch_mean], dim=1)
        logits = model.classifier(hidden_states)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return logits


def build_inputs_embeds(model, tokenizer, text, device):
    messages = [{"role": "user", "content": text}]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"].to(device)
    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        inputs_embeds = embed_layer(input_ids)
    return input_ids, inputs_embeds


def get_all_blocks_input_llm(model, inputs_embeds):
    target_dtype = model.get_input_embeddings().weight.dtype
    hidden_states = inputs_embeds.to(target_dtype)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
    
    block_inputs = []
    num_blocks = len(model.model.layers)
    
    with torch.no_grad():
        for idx in range(num_blocks):
            block_inputs.append(hidden_states.clone())
            output = model.model.layers[idx](hidden_states, position_ids=position_ids)
            hidden_states = output[0] if isinstance(output, tuple) else output
    
    return block_inputs


def get_all_blocks_input_image(model, model_type, image):
    block_inputs = []
    
    if model_type == 'vit_large':
        hidden_states = model.vit.embeddings(image)
        num_blocks = len(model.vit.encoder.layer)
        
        with torch.no_grad():
            for idx in range(num_blocks):
                block_inputs.append(hidden_states.clone())
                output = model.vit.encoder.layer[idx](hidden_states)
                hidden_states = output[0] if isinstance(output, tuple) else output
    
    elif model_type == 'dinov2_giant':
        hidden_states = model.dinov2.embeddings(image)
        num_blocks = len(model.dinov2.encoder.layer)
        
        with torch.no_grad():
            for idx in range(num_blocks):
                block_inputs.append(hidden_states.clone())
                output = model.dinov2.encoder.layer[idx](hidden_states)
                hidden_states = output[0] if isinstance(output, tuple) else output
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return block_inputs


def get_attack_block_indices(num_blocks, interval):
    if interval <= 0:
        return list(range(num_blocks))
    
    indices = []
    start_idx = interval - 1
    idx = start_idx
    while idx < num_blocks:
        indices.append(idx)
        idx += interval
    return indices


def get_no_token_id(tokenizer):
    for v in [" no", "No", " No", "NO", "no"]:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if len(ids) > 0:
            return ids[0]
    return None


def get_yes_token_ids(tokenizer):
    yes_ids = set()
    for v in [" yes", "Yes", " Yes", "YES", "yes", "Yes.", " yes.", " Yes."]:
        ids = tokenizer(v, add_special_tokens=False).input_ids
        if len(ids) > 0:
            yes_ids.add(ids[0])
    return yes_ids


def get_target_ids_llm(model, tokenizer, text, device, attack_mode):
    messages = [{"role": "user", "content": text}]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        base_out = model(input_ids=input_ids, use_cache=False)
        base_last = base_out.logits[:, -1, :].to(torch.float32)
        y_id = int(torch.argmax(base_last[0], dim=-1).item())
    
    if attack_mode == "targeted":
        n_id = get_no_token_id(tokenizer)
        if n_id is None:
            raise ValueError("Cannot find 'no' token")
        return y_id, n_id, None
    else:
        yes_ids = get_yes_token_ids(tokenizer)
        return y_id, None, yes_ids


def get_target_class_id_image(model, model_type, image, device):
    image = image.to(device)
    with torch.no_grad():
        logits = forward_with_all_blocks_attack_image(model, model_type, image, {}, [])
        if hasattr(logits, 'logits'):
            logits = logits.logits
        predicted_id = int(torch.argmax(logits, dim=-1).item())
    return predicted_id


def compute_margin_llm(last, y_id, n_id, yes_ids, attack_mode):
    if attack_mode == "targeted":
        return last[0, n_id] - last[0, y_id]
    else:
        yes_log = last[0, y_id]
        mask = torch.ones_like(last[0])
        mask[y_id] = 0.0
        if yes_ids:
            for yes_id in yes_ids:
                mask[yes_id] = 0.0
        masked_logits = last[0] * mask + (1.0 - mask) * (-1e9)
        max_other_smooth = torch.logsumexp(masked_logits, dim=0)
        return max_other_smooth - yes_log


def compute_margin_image(logits, orig_class_id, target_class_id, attack_mode):
    logits = logits.to(torch.float32)
    if hasattr(logits, 'logits'):
        logits = logits.logits
    if attack_mode == "targeted":
        return logits[0, target_class_id] - logits[0, orig_class_id]
    else:
        orig_log = logits[0, orig_class_id]
        mask = torch.ones_like(logits[0])
        mask[orig_class_id] = 0.0
        masked_logits = logits[0] * mask + (1.0 - mask) * (-1e9)
        max_other_smooth = torch.logsumexp(masked_logits, dim=0)
        return max_other_smooth - orig_log


def check_flipped_llm(last, y_id, n_id, yes_ids, attack_mode):
    top1 = torch.argmax(last[0], dim=-1).item()
    if attack_mode == "targeted":
        if n_id is None:
            return False
        return top1 == n_id
    else:
        if top1 == y_id:
            return False
        if yes_ids and top1 in yes_ids:
            return False
        return True


def check_flipped_image(logits, orig_class_id, target_class_id, attack_mode):
    if hasattr(logits, 'logits'):
        logits = logits.logits
    top1 = torch.argmax(logits[0], dim=-1).item()
    if attack_mode == "targeted":
        return top1 == target_class_id
    else:
        return top1 != orig_class_id


def project_delta(delta, epsilon):
    if len(delta.shape) == 3:
        d_norm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), ord=2, dim=1, keepdim=True).view(-1, 1, 1) + 1e-12
    else:
        d_norm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), ord=2, dim=1, keepdim=True) + 1e-12
    return delta * torch.clamp(epsilon / d_norm, max=1.0)


def init_delta(x0, epsilon):
    z = torch.randn_like(x0)
    if len(z.shape) == 3:
        z_norm = torch.linalg.vector_norm(z.reshape(z.shape[0], -1), ord=2, dim=1, keepdim=True).view(-1, 1, 1) + 1e-12
    else:
        z_norm = torch.linalg.vector_norm(z.reshape(z.shape[0], -1), ord=2, dim=1, keepdim=True) + 1e-12
    return (z / z_norm) * (torch.rand(1, device=x0.device) * epsilon)


def pgd_attack_all_blocks_llm(model, tokenizer, text, device, block_epsilons,
                              iters, step_size, attack_mode, y_id,
                              n_id, yes_ids, interval=0, use_checkpoint=False):
    input_ids, inputs_embeds = build_inputs_embeds(model, tokenizer, text, device)
    block_inputs = get_all_blocks_input_llm(model, inputs_embeds)
    num_blocks = len(block_inputs)
    
    attack_indices = get_attack_block_indices(num_blocks, interval)
    
    block_x0s = {idx: block_inputs[idx].detach().to(torch.float32) for idx in attack_indices}
    block_deltas = {idx: init_delta(block_x0s[idx], block_epsilons[idx]) for idx in attack_indices}
    block_xs = {idx: block_x0s[idx] + block_deltas[idx] for idx in attack_indices}
    
    for t in range(iters):
        block_xs_grad = {idx: block_xs[idx].detach().clone().requires_grad_(True) for idx in attack_indices}
        current_deltas = {idx: block_xs_grad[idx] - block_x0s[idx] for idx in attack_indices}
        
        torch.cuda.empty_cache()
        outputs = forward_with_all_blocks_attack_llm(model, inputs_embeds, current_deltas, attack_indices, use_checkpoint=use_checkpoint)
        last = outputs.logits[:, -1, :].to(torch.float32)
        
        if check_flipped_llm(last, y_id, n_id, yes_ids, attack_mode):
            del block_xs_grad, current_deltas, outputs, last
            break
        
        loss = compute_margin_llm(last, y_id, n_id, yes_ids, attack_mode)
        loss.backward()
        
        new_block_deltas = {}
        for idx in attack_indices:
            grad = block_xs_grad[idx].grad.detach()
            step = max(step_size, 1e-6)
            
            if len(grad.shape) == 3:
                g_norm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), ord=2, dim=1, keepdim=True).view(-1, 1, 1) + 1e-12
            else:
                g_norm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), ord=2, dim=1, keepdim=True) + 1e-12
            
            x_new = block_xs_grad[idx] + step * (grad / g_norm)
            
            delta = project_delta(x_new - block_x0s[idx], block_epsilons[idx])
            new_block_deltas[idx] = delta
        
        del block_xs_grad, current_deltas, outputs, last, loss
        torch.cuda.empty_cache()
        
        block_xs = {idx: block_x0s[idx] + new_block_deltas[idx] for idx in attack_indices}
        block_deltas = new_block_deltas
    
    with torch.no_grad():
        adv_out = forward_with_all_blocks_attack_llm(model, inputs_embeds, block_deltas, attack_indices)
        adv_last = adv_out.logits[:, -1, :].to(torch.float32)
        flipped = check_flipped_llm(adv_last, y_id, n_id, yes_ids, attack_mode)
        
        block_l2s = {idx: torch.norm(delta).item() for idx, delta in block_deltas.items()}
    
    return flipped, block_deltas, block_l2s


def pgd_attack_all_blocks_image(model, model_type, image, device, block_epsilons,
                                iters, step_size, orig_class_id, target_class_id,
                                attack_mode, interval=0, use_checkpoint=False):
    image = image.to(device)
    block_inputs = get_all_blocks_input_image(model, model_type, image)
    num_blocks = len(block_inputs)
    
    attack_indices = get_attack_block_indices(num_blocks, interval)
    
    block_x0s = {idx: block_inputs[idx].detach().to(torch.float32) for idx in attack_indices}
    block_deltas = {idx: init_delta(block_x0s[idx], block_epsilons[idx]) for idx in attack_indices}
    block_xs = {idx: block_x0s[idx] + block_deltas[idx] for idx in attack_indices}
    
    for t in range(iters):
        block_xs_grad = {idx: block_xs[idx].detach().clone().requires_grad_(True) for idx in attack_indices}
        current_deltas = {idx: block_xs_grad[idx] - block_x0s[idx] for idx in attack_indices}
        
        torch.cuda.empty_cache()
        logits = forward_with_all_blocks_attack_image(model, model_type, image, current_deltas, attack_indices, use_checkpoint=use_checkpoint)
        logits = logits.to(torch.float32)
        
        if check_flipped_image(logits, orig_class_id, target_class_id, attack_mode):
            del block_xs_grad, current_deltas, logits
            break
        
        loss = compute_margin_image(logits, orig_class_id, target_class_id, attack_mode)
        loss.backward()
        
        new_block_deltas = {}
        for idx in attack_indices:
            grad = block_xs_grad[idx].grad.detach()
            step = max(step_size, 1e-6)
            
            if len(grad.shape) == 3:
                g_norm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), ord=2, dim=1, keepdim=True).view(-1, 1, 1) + 1e-12
            else:
                g_norm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), ord=2, dim=1, keepdim=True) + 1e-12
            
            x_new = block_xs_grad[idx] + step * (grad / g_norm)
            
            delta = project_delta(x_new - block_x0s[idx], block_epsilons[idx])
            new_block_deltas[idx] = delta
        
        del block_xs_grad, current_deltas, logits, loss
        torch.cuda.empty_cache()
        
        block_xs = {idx: block_x0s[idx] + new_block_deltas[idx] for idx in attack_indices}
        block_deltas = new_block_deltas
    
    with torch.no_grad():
        adv_logits = forward_with_all_blocks_attack_image(model, model_type, image, block_deltas, attack_indices)
        adv_logits = adv_logits.to(torch.float32)
        flipped = check_flipped_image(adv_logits, orig_class_id, target_class_id, attack_mode)
        
        block_l2s = {idx: torch.norm(delta).item() for idx, delta in block_deltas.items()}
    
    return flipped, block_deltas, block_l2s


def compute_norms_all_blocks(deltas, block_inputs):
    if not deltas:
        return 0.0, {}, {}
    
    total_linf = max(delta.abs().max().item() for delta in deltas.values())
    block_l2s = {idx: torch.norm(delta).item() for idx, delta in deltas.items()}
    block_rel_l2s = {
        idx: block_l2s[idx] / (torch.norm(block_inputs[idx]).item() + 1e-10)
        for idx in deltas.keys()
    }
    return total_linf, block_l2s, block_rel_l2s


def format_block_l2s(block_l2s, block_rel_l2s):
    if not block_rel_l2s:
        return ""
    
    parts = []
    for idx in sorted(block_rel_l2s.keys()):
        rel_l2 = block_rel_l2s[idx]
        parts.append(f"block_{idx}: L2={rel_l2:.3e}")
    
    return " | ".join(parts)


def format_scientific_latex(value):
    if value == 0.0:
        return "$0$"
    
    s = f"{value:.3e}"
    if 'e' in s:
        coeff, exp = s.split('e')
        coeff = coeff.rstrip('0').rstrip('.')
        exp_int = int(exp)
        if exp_int == 0:
            return f"${coeff}$"
        else:
            return f"${coeff} \\times 10^{{{exp_int}}}$"
    return f"${s}$"


def show_attack_result_llm(model, tokenizer, text, device, attack_type,
                          attack_location, delta_or_deltas,
                          attack_mode="untargeted"):
    input_ids, inputs_embeds = build_inputs_embeds(model, tokenizer, text, device)
    
    with torch.no_grad():
        original_outputs = forward_with_all_blocks_attack_llm(model, inputs_embeds, {}, [])
        original_logits = original_outputs.logits[:, -1, :]
        original_top_id = torch.argmax(original_logits[0], dim=-1).item()
        original_top_token = tokenizer.decode([original_top_id], skip_special_tokens=True)
    
    with torch.no_grad():
        attack_indices = list(delta_or_deltas.keys())
        outputs = forward_with_all_blocks_attack_llm(model, inputs_embeds, delta_or_deltas, attack_indices)
        
        logits = outputs.logits[:, -1, :]
        top_token_id = torch.argmax(logits[0], dim=-1).item()
        top_token = tokenizer.decode([top_token_id], skip_special_tokens=True)
        
        print(f"Before: {original_top_token}")
        print(f"After: {top_token}")


def show_attack_result_image(model, model_type, image, device, attack_type,
                            attack_location, delta_or_deltas, orig_class_id, target_class_id,
                            attack_mode):
    image = image.to(device)
    
    with torch.no_grad():
        original_logits = forward_with_all_blocks_attack_image(model, model_type, image, {}, [])
        if hasattr(original_logits, 'logits'):
            original_logits = original_logits.logits
        original_logits = original_logits.to(torch.float32)
        original_top_id = torch.argmax(original_logits[0], dim=-1).item()
    
    with torch.no_grad():
        attack_indices = list(delta_or_deltas.keys())
        logits = forward_with_all_blocks_attack_image(model, model_type, image, delta_or_deltas, attack_indices)
        if hasattr(logits, 'logits'):
            logits = logits.logits
        logits = logits.to(torch.float32)
        top_class_id = torch.argmax(logits[0], dim=-1).item()
        
        print(f"Before: class {orig_class_id}")
        print(f"After: class {top_class_id}")


def run_single_sample(model, device, max_epsilon, pgd_iters, interval=0, 
                      use_checkpoint=False, search_steps=32, retry_times=2,
                      tokenizer=None, text=None, attack_mode=None,
                      model_type=None, image=None, target_class_id=None):
    is_image_mode = model_type is not None
    
    if is_image_mode:
        image = image.to(device)
        orig_class_id = get_target_class_id_image(model, model_type, image, device)
        block_inputs = get_all_blocks_input_image(model, model_type, image)
    else:
        _, original_embeds = build_inputs_embeds(model, tokenizer, text, device)
        y_id, n_id, yes_ids = get_target_ids_llm(model, tokenizer, text, device, attack_mode)
        if y_id is None:
            raise ValueError("Cannot find target token")
        block_inputs = get_all_blocks_input_llm(model, original_embeds)
    
    num_blocks = len(block_inputs)
    attack_indices = get_attack_block_indices(num_blocks, interval)
    
    def attack_fn(block_epsilons, step_size):
        if is_image_mode:
            return pgd_attack_all_blocks_image(
                model, model_type, image, device, block_epsilons, pgd_iters, step_size,
                orig_class_id, target_class_id, attack_mode, interval, use_checkpoint=use_checkpoint
            )
        else:
            return pgd_attack_all_blocks_llm(
                model, tokenizer, text, device, block_epsilons, pgd_iters, step_size,
                attack_mode, y_id, n_id, yes_ids, interval, use_checkpoint=use_checkpoint
            )
    
    block_epsilons_max = {idx: max_epsilon for idx in attack_indices}
    flipped_max, delta_max, block_l2s_max = attack_fn(block_epsilons_max, max_epsilon * 0.25)
    
    if not flipped_max:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    block_lo = {idx: 0.0 for idx in attack_indices}
    block_hi = {idx: max_epsilon for idx in attack_indices}
    best_block_epsilons = {idx: max_epsilon for idx in attack_indices}
    best_delta = delta_max
    best_block_l2s = block_l2s_max
    
    pbar = tqdm(range(search_steps), desc="Binary search", leave=False)
    for step_idx in pbar:
        block_epsilons_mid = {idx: (block_lo[idx] + block_hi[idx]) / 2.0 for idx in attack_indices}
        step_size = max_epsilon * 0.25
        
        flipped = False
        delta = None
        block_l2s = None
        for attempt in range(retry_times):
            flipped, delta, block_l2s = attack_fn(block_epsilons_mid, step_size)
            if flipped:
                break
        
        if flipped:
            best_block_epsilons = block_epsilons_mid.copy()
            best_delta = delta
            best_block_l2s = block_l2s
            for idx in attack_indices:
                block_hi[idx] = block_epsilons_mid[idx]
            min_actual_eps = min(best_block_l2s.values()) if best_block_l2s else 0.0
            pbar.set_postfix({"min_eps": f"{min_actual_eps:.3e}"})
        else:
            for idx in attack_indices:
                block_lo[idx] = block_epsilons_mid[idx]
    
    if best_delta is None:
        best_delta = delta_max
        best_block_l2s = block_l2s_max
    
    linf, block_l2s, block_rel_l2s = compute_norms_all_blocks(best_delta, block_inputs)
    max_eps = max(best_block_epsilons.values()) if best_block_epsilons else max_epsilon
    pgd_result = (max_eps, linf, block_l2s, block_rel_l2s, "all_blocks", best_delta)
    
    return pgd_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_path', type=str, default="./finetuned_model", help='Model path')
    parser.add_argument("--model_type", type=str, default=None,
                       choices=[None, "vit_large", "dinov2_giant"])
    parser.add_argument('-d', '--device', type=str, default="cuda:0")
    parser.add_argument("--max_epsilon", type=float, default=1.0)
    parser.add_argument("--pgd_iters", type=int, default=40)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--image_path", type=str, default="image.jpg")
    parser.add_argument("--target_class_id", type=int, default=0)
    parser.add_argument("--attack_mode", type=str, default="untargeted", choices=["targeted", "untargeted"])
    parser.add_argument("--interval", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--retry_times", type=int, default=2)
    args = parser.parse_args()

    is_image_mode = args.model_type is not None

    if is_image_mode:
        model = load_image_model(args.model_path, args.model_type, args.device)
        device = next(model.parameters()).device

        image = load_image(args.image_path)
        
        if args.interval > 0:
            if args.model_type == 'vit_large':
                num_blocks = len(model.vit.encoder.layer)
            elif args.model_type == 'dinov2_giant':
                num_blocks = len(model.dinov2.encoder.layer)
            else:
                num_blocks = 0
            attack_indices = get_attack_block_indices(num_blocks, args.interval)
        
        target_class_id = args.target_class_id if args.attack_mode == "targeted" else None
        pgd_result = run_single_sample(
            model, device,
            args.max_epsilon, args.pgd_iters,
            interval=args.interval,
            use_checkpoint=args.use_checkpoint,
            retry_times=args.retry_times,
            model_type=args.model_type, image=image, target_class_id=target_class_id, attack_mode=args.attack_mode
        )
        
        if pgd_result is not None:
            eps, linf, l2_or_blocks, rel_l2_or_blocks, attack_location, delta_or_deltas = pgd_result
            print(f"PGD: epsilon={eps:.3e}, L_inf={linf:.3e}, location={attack_location}")
            
            orig_class_id = get_target_class_id_image(model, args.model_type, image, device)
            show_attack_result_image(model, args.model_type, image, device, "PGD", attack_location, delta_or_deltas, 
                                   orig_class_id, target_class_id, args.attack_mode)
            
    else:
        model, tokenizer = load_base_model(args.model_path, args.device)
        device = next(model.parameters()).device

        text = args.text or "Answer with yes or no only. Is 2+2 equal to 4?"
        
        if args.interval > 0:
            num_blocks = len(model.model.layers)
            attack_indices = get_attack_block_indices(num_blocks, args.interval)
        
        pgd_result = run_single_sample(
            model, device,
            args.max_epsilon, args.pgd_iters,
            interval=args.interval,
            use_checkpoint=args.use_checkpoint,
            retry_times=args.retry_times,
            tokenizer=tokenizer, text=text, attack_mode=args.attack_mode
        )
        
        if pgd_result is not None:
            eps, linf, l2_or_blocks, rel_l2_or_blocks, attack_location, delta_or_deltas = pgd_result
            print(f"PGD: epsilon={eps:.3e}, L_inf={linf:.3e}, location={attack_location}")
            show_attack_result_llm(model, tokenizer, text, device, "PGD", attack_location, delta_or_deltas, attack_mode=args.attack_mode)
            


if __name__ == "__main__":
    main()
