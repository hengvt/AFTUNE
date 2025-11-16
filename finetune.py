import torch
import torch.optim as optim
import torch.nn as nn
import time
from tqdm import tqdm

def register_module_hooks(module, module_name, recorder, is_first_layer, is_last_layer):
    forward_hook = module.register_forward_hook(
        recorder.create_forward_hook(module_name, is_first_layer=is_first_layer)
    )
    backward_hook = module.register_full_backward_hook(
        recorder.create_backward_hook(module_name, is_last_layer=is_last_layer)
    )
    return (forward_hook, backward_hook)

def get_tracked_modules(model):
    tracked_modules = {}
    num_layers = len(model.model.layers)
    
    tracked_modules["embedding"] = model.model.embed_tokens
    for idx in range(num_layers):
        tracked_modules[f"layer_{idx}"] = model.model.layers._modules[str(idx)]
    if hasattr(model.model, 'norm'):
        tracked_modules["norm"] = model.model.norm
    tracked_modules["lm_head"] = model.lm_head
    
    return tracked_modules

def register_hooks_with_recorder(model, recorder):
    hooks = []
    num_layers = len(model.model.layers)
    
    embed_forward = model.model.embed_tokens.register_forward_hook(
        recorder.create_forward_hook("embedding", is_first_layer=True)
    )
    embed_backward = model.model.embed_tokens.register_full_backward_hook(
        recorder.create_backward_hook("embedding")
    )
    hooks.append((embed_forward, embed_backward))
    
    for idx in range(num_layers):
        layer_name = f"layer_{idx}"
        layer = model.model.layers._modules[str(idx)]
        
        forward_hook = layer.register_forward_hook(
            recorder.create_forward_hook(layer_name)
        )
        backward_hook = layer.register_full_backward_hook(
            recorder.create_backward_hook(layer_name)
        )
        
        hooks.append((forward_hook, backward_hook))
    
    if hasattr(model.model, 'norm'):
        norm_forward = model.model.norm.register_forward_hook(
            recorder.create_forward_hook("norm")
        )
        norm_backward = model.model.norm.register_full_backward_hook(
            recorder.create_backward_hook("norm")
        )
        hooks.append((norm_forward, norm_backward))
    
    lm_head_forward = model.lm_head.register_forward_hook(
        recorder.create_forward_hook("lm_head", is_last_layer=True)
    )
    lm_head_backward = model.lm_head.register_full_backward_hook(
        recorder.create_backward_hook("lm_head", is_last_layer=True)
    )
    hooks.append((lm_head_forward, lm_head_backward))
    
    return hooks

def finetune_llm_full(model, dataloader, recorder, epochs, learning_rate, max_samples, warmup_steps):
    model.train()
    
    tracked_modules = get_tracked_modules(model)
    
    recorder.layer_order = list(tracked_modules.keys())
    recorder.layer_order_finalized = True
    recorder.finalize_layer_blocks()
    
    for layer_name, module in tracked_modules.items():
        recorder.save_module_structure(layer_name, module)
    
    hooks = register_hooks_with_recorder(model, recorder)
    
    if recorder.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    device = model.device
    
    recorder.learning_rate = learning_rate
    recorder.set_optimizer(optimizer)
    
    if warmup_steps > 0:
        recorder.set_warmup_mode(True)
        
        warmup_count = 0
        for batch in dataloader:
            if warmup_count >= warmup_steps:
                break
            
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch, use_cache=False)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmup_count += 1
        
        recorder.set_warmup_mode(False)
        recorder.current_step = 0
        recorder.records.clear()
        recorder.dataset_records.clear()
        recorder.clear_time_stats()
    
    total_samples = 0
    max_steps = None
    if max_samples is not None:
        max_steps = (max_samples + dataloader.batch_size - 1) // dataloader.batch_size
    
    total_steps = 0
    
    total_batches = len(dataloader) * epochs if max_steps is None else max_steps
    
    pbar = tqdm(total=total_batches, desc="Training", unit="step")
    
    train_start = time.time()
    for epoch in range(epochs):
        for batch in dataloader:
            if max_steps is not None and recorder.current_step >= max_steps:
                break
            
            for layer_name, module in tracked_modules.items():
                recorder.record_checkpoint(layer_name, module)
            
            batch = batch.to(device)
            
            outputs = model(input_ids=batch, labels=batch, use_cache=False)
            loss = outputs.loss
            
            recorder.record_dataset_input(batch, batch, loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_steps += 1
            
            recorder.increment_step()
            total_samples += batch.shape[0]
            
            pbar.update(1)
            pbar.set_postfix({
                'epoch': f'{epoch+1}/{epochs}',
                'loss': f'{loss.item():.4f}',
                'samples': total_samples
            })
        
        if max_steps is not None and recorder.current_step >= max_steps:
            break
    train_end = time.time()
    total_train_time = train_end - train_start
    
    pbar.close()
    
    recorder.wait_all_tasks()
    
    for forward_hook, backward_hook in hooks:
        forward_hook.remove()
        if backward_hook is not None:
            backward_hook.remove()
    
    if total_steps > 0:
        print(f"\nTraining statistics: {total_steps} steps, {total_samples} samples, {total_train_time:.3f} seconds")





class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, device=None, dtype=None):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float32
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank, device=device, dtype=dtype) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim, device=device, dtype=dtype))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        device = next(linear.parameters()).device
        dtype = next(linear.parameters()).dtype
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, device=device, dtype=dtype
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def apply_lora_to_model(model, target_modules, lora_r, lora_alpha):
    lora_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = name.split('.')[-1]
            if module_name in target_modules:
                parent = model
                path = name.split('.')
                for p in path[:-1]:
                    parent = getattr(parent, p)
                
                lora_module = LinearWithLoRA(module, lora_r, lora_alpha)
                setattr(parent, path[-1], lora_module)
                lora_modules[name] = lora_module
    
    return lora_modules

def finetune_llm_lora(model, dataloader, recorder, epochs, learning_rate, 
                  max_samples, target_modules, lora_r, lora_alpha, warmup_steps):    
    model.train()
    
    lora_modules = apply_lora_to_model(model, target_modules, lora_r, lora_alpha)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for lora_module in lora_modules.values():
        for param in lora_module.lora.parameters():
            param.requires_grad = True
    
    tracked_modules = get_tracked_modules(model)
    
    recorder.layer_order = list(tracked_modules.keys())
    recorder.layer_order_finalized = True
    recorder.finalize_layer_blocks()
    
    recorder.use_lora = True
    recorder.lora_modules = lora_modules
    
    for layer_name, module in tracked_modules.items():
        recorder.save_module_structure(layer_name, module)
    
    hooks = register_hooks_with_recorder(model, recorder)
    
    lora_params = []
    for lora_module in lora_modules.values():
        lora_params.extend(lora_module.lora.parameters())
    
    if recorder.optimizer_type == 'sgd':
        optimizer = optim.SGD(lora_params, lr=learning_rate)
    else:
        optimizer = optim.AdamW(lora_params, lr=learning_rate)
    
    device = model.device
    
    recorder.learning_rate = learning_rate
    recorder.set_optimizer(optimizer)
    
    if warmup_steps > 0:
        recorder.set_warmup_mode(True)
        
        warmup_count = 0
        for batch in dataloader:
            if warmup_count >= warmup_steps:
                break
            
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch, use_cache=False)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmup_count += 1
        
        recorder.set_warmup_mode(False)
        recorder.current_step = 0
        recorder.records.clear()
        recorder.dataset_records.clear()
        recorder.clear_time_stats()
    
    total_samples = 0
    max_steps = None
    if max_samples is not None:
        max_steps = (max_samples + dataloader.batch_size - 1) // dataloader.batch_size
    
    total_steps = 0
    
    total_batches = len(dataloader) * epochs if max_steps is None else max_steps
    
    pbar = tqdm(total=total_batches, desc="LoRA training", unit="step")
    
    train_start = time.time()
    for epoch in range(epochs):
        for batch in dataloader:
            if max_steps is not None and recorder.current_step >= max_steps:
                break
            
            for layer_name, module in tracked_modules.items():
                recorder.record_checkpoint(layer_name, module)
            
            batch = batch.to(device)
            
            outputs = model(input_ids=batch, labels=batch, use_cache=False)
            loss = outputs.loss
            
            recorder.record_dataset_input(batch, batch, loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_steps += 1
            
            recorder.increment_step()
            total_samples += batch.shape[0]
            
            pbar.update(1)
            pbar.set_postfix({
                'epoch': f'{epoch+1}/{epochs}',
                'loss': f'{loss.item():.4f}',
                'samples': total_samples
            })
        
        if max_steps is not None and recorder.current_step >= max_steps:
            break
    train_end = time.time()
    total_train_time = train_end - train_start
    
    pbar.close()
    
    recorder.wait_all_tasks()
    
    for forward_hook, backward_hook in hooks:
        forward_hook.remove()
        if backward_hook is not None:
            backward_hook.remove()
    
    if total_steps > 0:
        print(f"\nLoRA training statistics: {total_steps} steps, {total_samples} samples, {len(lora_modules)} modules, {total_train_time:.3f} seconds")


def dinov2_forward(model, inputs, tracked_modules, loss_fn, labels):
    embedding_module = tracked_modules['embeddings']
    
    embedding_output = embedding_module(inputs)
    embedding_output.retain_grad()
    
    x = embedding_output
    for i in range(len(model.dinov2.encoder.layer)):
        layer_output = tracked_modules[f'encoder_layer_{i}'](x)
        if isinstance(layer_output, tuple):
            x = layer_output[0]
        else:
            x = layer_output
    x = tracked_modules['layernorm'](x)
    
    if len(x.shape) == 3:
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        patch_mean = patch_tokens.mean(dim=1)
        processed_input = torch.cat([cls_token, patch_mean], dim=1)
    else:
        processed_input = x
    logits = tracked_modules['classifier'](processed_input)
    
    if labels is not None:
        logits = logits.to(torch.float32)
        loss = loss_fn(logits, labels)
    else:
        loss = None
    
    return embedding_output, loss

def dinov2_backward(model, embedding_output, loss):
    non_embedding_params = []
    for name, param in model.named_parameters():
        if 'embeddings' not in name and param.requires_grad:
            non_embedding_params.append(param)
    
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=[embedding_output] + non_embedding_params,
        retain_graph=False,
        create_graph=False
    )
    
    embedding_output_grad = grads[0]
    
    for param, grad in zip(non_embedding_params, grads[1:]):
        param.grad = grad
    
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        torch.autograd.backward(embedding_output, embedding_output_grad)
    finally:
        torch.use_deterministic_algorithms(prev)

def finetune_imagenet(model, dataloader, recorder, model_hooks, epochs, learning_rate, max_samples, warmup_steps):
    model.train()
    
    loss_fn = nn.CrossEntropyLoss()
    
    tracked_modules = {name: module for module, name in model_hooks}
    
    recorder.layer_order = [name for _, name in model_hooks]
    recorder.layer_order_finalized = True
    recorder.finalize_layer_blocks()
    
    for layer_name, module in tracked_modules.items():
        recorder.save_module_structure(layer_name, module)
    
    hooks = []
    for idx, (module, layer_name) in enumerate(model_hooks):
        is_first = (idx == 0)
        is_last = (idx == len(model_hooks) - 1)
        hook_pair = register_module_hooks(module, layer_name, recorder, is_first_layer=is_first, is_last_layer=is_last)
        hooks.append(hook_pair)
    
    if recorder.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    device = next(model.parameters()).device
    
    recorder.learning_rate = learning_rate
    recorder.set_optimizer(optimizer)
    
    if warmup_steps > 0:
        recorder.set_warmup_mode(True)
        
        warmup_count = 0
        for batch_data in dataloader:
            if warmup_count >= warmup_steps:
                break
            
            if isinstance(batch_data, (list, tuple)):
                inputs, labels = batch_data
            else:
                inputs = batch_data
                labels = None
            
            inputs = inputs.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            model_dtype = next(model.parameters()).dtype
            if inputs.dtype != torch.long:
                inputs = inputs.to(dtype=model_dtype)
            
            is_dinov2_giant = (recorder.model_name == 'dinov2_giant')
            
            if is_dinov2_giant:
                embedding_output, loss = dinov2_forward(model, inputs, tracked_modules, loss_fn, labels)
            else:
                outputs = model(inputs)
                
                if labels is not None:
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('loss'))
                        if logits is not None:
                            logits = logits.to(torch.float32)
                            loss = loss_fn(logits, labels)
                        else:
                            loss = outputs.get('loss')
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits.to(torch.float32)
                        loss = loss_fn(logits, labels)
                    else:
                        logits = outputs.to(torch.float32) if isinstance(outputs, torch.Tensor) else outputs
                        loss = loss_fn(logits, labels)
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else None
            
            if loss is not None:
                optimizer.zero_grad()
                
                if is_dinov2_giant:
                    dinov2_backward(model, embedding_output, loss)
                else:
                    loss.backward()
                
                optimizer.step()
            
            warmup_count += 1
        
        recorder.set_warmup_mode(False)
        recorder.current_step = 0
        recorder.records.clear()
        recorder.dataset_records.clear()
        recorder.clear_time_stats()
    
    total_samples = 0
    max_steps = (max_samples + dataloader.batch_size - 1) // dataloader.batch_size
    
    total_steps = 0
    
    total_batches = max_steps * epochs
    
    pbar = tqdm(total=total_batches, desc="Training", unit="step")
    
    train_start = time.time()
    for epoch in range(epochs):
        for batch_data in dataloader:
            if recorder.current_step >= max_steps:
                break
            
            if isinstance(batch_data, (list, tuple)):
                inputs, labels = batch_data
            else:
                inputs = batch_data
                labels = None
            
            for layer_name, module in tracked_modules.items():
                recorder.record_checkpoint(layer_name, module)
            
            inputs = inputs.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            model_dtype = next(model.parameters()).dtype
            if inputs.dtype != torch.long:
                inputs = inputs.to(dtype=model_dtype)
            
            is_dinov2_giant = (recorder.model_name == 'dinov2_giant')
            
            if is_dinov2_giant:
                embedding_output, loss = dinov2_forward(model, inputs, tracked_modules, loss_fn, labels)
            else:
                outputs = model(inputs)
                
                if labels is not None:
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('loss'))
                        if logits is not None:
                            logits = logits.to(torch.float32)
                            loss = loss_fn(logits, labels)
                        else:
                            loss = outputs.get('loss')
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits.to(torch.float32)
                        loss = loss_fn(logits, labels)
                    else:
                        logits = outputs.to(torch.float32) if isinstance(outputs, torch.Tensor) else outputs
                        loss = loss_fn(logits, labels)
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else None
            
            recorder.record_dataset_input(inputs, labels, loss)
            optimizer.zero_grad()
            
            if is_dinov2_giant:
                dinov2_backward(model, embedding_output, loss)
            else:
                loss.backward()
            
            optimizer.step()
            
            total_steps += 1
            
            recorder.increment_step()
            batch_size = inputs.shape[0] if isinstance(inputs, torch.Tensor) else len(inputs)
            total_samples += batch_size
            
            pbar.update(1)
            loss_val = loss.item() if loss is not None else 0.0
            pbar.set_postfix({
                'epoch': f'{epoch+1}/{epochs}',
                'loss': f'{loss_val:.4f}',
                'samples': total_samples
            })
        
        if recorder.current_step >= max_steps:
            break
    train_end = time.time()
    total_train_time = train_end - train_start
    
    pbar.close()
    
    recorder.wait_all_tasks()
    
    for forward_hook, backward_hook in hooks:
        forward_hook.remove()
        if backward_hook is not None:
            backward_hook.remove()
    
    if total_steps > 0:
        print(f"\nTraining statistics: {total_steps} steps, {total_samples} samples, {total_train_time:.3f} seconds")