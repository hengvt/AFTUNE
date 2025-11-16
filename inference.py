import torch
import argparse
import os
import shutil
import hashlib
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from layer_recorder import LayerRecorder, set_deterministic
from finetune import get_tracked_modules
from load_model import load_resnet152, load_vit_large, load_dinov2_giant
from PIL import Image
from torchvision import transforms


def get_dir_size(dir_path):
    total_size = 0
    if not os.path.exists(dir_path):
        return 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


def format_size(size_bytes):
    for unit in ['B', 'KB']:
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} MB"


def record_inference_llm(model, tokenizer, prompt, recorder, device):
    model.eval()
    
    tracked_modules = get_tracked_modules(model)
    recorder.layer_order = list(tracked_modules.keys())
    recorder.layer_order_finalized = True
    recorder.finalize_layer_blocks()
    
    hooks = []
    num_layers = len(model.model.layers)
    
    embed_forward = model.model.embed_tokens.register_forward_hook(
        recorder.create_forward_hook("embedding", is_first_layer=True)
    )
    hooks.append(embed_forward)
    
    for idx in range(num_layers):
        layer_name = f"layer_{idx}"
        layer = model.model.layers._modules[str(idx)]
        forward_hook = layer.register_forward_hook(
            recorder.create_forward_hook(layer_name)
        )
        hooks.append(forward_hook)
    
    if hasattr(model.model, 'norm'):
        norm_forward = model.model.norm.register_forward_hook(
            recorder.create_forward_hook("norm")
        )
        hooks.append(norm_forward)
    
    messages = [{"role": "user", "content": prompt}]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"].to(device)
    
    inference_start = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits
    inference_time = time.time() - inference_start
    predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=False)
    
    if recorder.current_step not in recorder.dataset_records:
        recorder.dataset_records[recorder.current_step] = {}
    recorder.dataset_records[recorder.current_step]['label'] = predicted_token_id
    recorder.dataset_records[recorder.current_step]['input_ids'] = input_ids.cpu()
    
    if recorder.async_save and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    recorder.save_all_records()
    
    
    for hook in hooks:
        hook.remove()
    
    
    return inference_time

def record_inference_image(model, model_hooks, image, recorder, device, model_name):
    model.eval()
    
    tracked_modules = {name: module for module, name in model_hooks}
    recorder.layer_order = [name for _, name in model_hooks]
    recorder.layer_order_finalized = True
    recorder.finalize_layer_blocks()
    
    hooks = []
    for idx, (module, layer_name) in enumerate(model_hooks):
        is_first = (idx == 0)
        forward_hook = module.register_forward_hook(
            recorder.create_forward_hook(layer_name, is_first_layer=is_first)
        )
        hooks.append(forward_hook)
    
    image = image.to(device)
    model_dtype = next(model.parameters()).dtype
    if image.dtype != torch.long:
        image = image.to(dtype=model_dtype)
    
    inference_start = time.time()
    with torch.no_grad():
        outputs = model(image)
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs)
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
    inference_time = time.time() - inference_start
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    if recorder.current_step not in recorder.dataset_records:
        recorder.dataset_records[recorder.current_step] = {}
    recorder.dataset_records[recorder.current_step]['label'] = predicted_label
    
    if recorder.async_save and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    recorder.save_all_records()
    
    
    for hook in hooks:
        hook.remove()
    
    
    return inference_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_path', type=str, default='./finetuned_model', help='Model path')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-r', '--record_dir', type=str, default='./inference_records')
    parser.add_argument('--prompt', type=str, default="Answer with yes or no only. Is 2+2 equal to 4?",
                       help='LLM input text')
    parser.add_argument('--image_path', type=str, default='image.jpg',
                       help='Image file path')
    parser.add_argument('-l', '--layers_per_block', type=int, default=1, help='Layers per block (B_L)')
    parser.add_argument('--activation_interval', type=int, default=1, help='Activation save interval (I_A)')
    parser.add_argument('--disabled', action='store_true', help='Disable AFTUNE')
    parser.add_argument('--module_structure_dir', type=str, default='./module_structures',
                       help='Module structure directory')
    parser.add_argument('--block_records_dir', type=str, default='./block_records',
                       help='Training record directory')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_float32', action='store_true', help='Load model as float32')
    parser.add_argument('--use_float64', action='store_true', help='Load model as float64')
    args = parser.parse_args()
    
    set_deterministic(seed=args.seed, enabled=args.deterministic)
    
    block_recorder = LayerRecorder(save_dir=args.block_records_dir)
    block_metadata = block_recorder.load_metadata()
    if block_metadata and 'model_name' in block_metadata:
        model_name = block_metadata['model_name']
        chunk_size = block_metadata['chunk_size']
        use_blake3 = block_metadata['use_blake3']
    else:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        chunk_size = 4096
        use_blake3 = True
    
    is_llm = model_name not in ['resnet152', 'vit_large', 'dinov2_giant']
    
    model = None
    tokenizer = None
    model_hooks = None
    
    if is_llm:
        from load_model import load_llm_model
        model, tokenizer = load_llm_model(args.model_path, args.device)
        
        if args.use_float64:
            model = model.to(dtype=torch.float64)
        elif args.use_float32:
            model = model.to(dtype=torch.float32)
        model.eval()
        
        input_hash = hashlib.sha256(args.prompt.encode()).hexdigest()[:16]
    else:
        if model_name == 'resnet152':
            model, model_hooks = load_resnet152(pretrained=False, device=args.device)
            state_dict_path = os.path.join(args.model_path, 'resnet152.pth')
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location=args.device, weights_only=False)
                model.load_state_dict(state_dict)
                if args.use_float64:
                    model = model.to(dtype=torch.float64)
                elif args.use_float32:
                    model = model.to(dtype=torch.float32)
            else:
                raise FileNotFoundError(f"Model file not found: {state_dict_path}")
        elif model_name == 'vit_large':
            model, model_hooks = load_vit_large(device=args.device)
            state_dict_path = os.path.join(args.model_path, 'vit_large.pth')
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location=args.device, weights_only=False)
                model.load_state_dict(state_dict)
            if args.use_float64:
                model = model.to(dtype=torch.float64)
            elif args.use_float32:
                model = model.to(dtype=torch.float32)
        elif model_name == 'dinov2_giant':
            model, model_hooks = load_dinov2_giant(pretrained=True, device=args.device)
            state_dict_path = os.path.join(args.model_path, 'dinov2_giant.pth')
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location=args.device, weights_only=False)
                model.load_state_dict(state_dict)
                if args.use_float64:
                    model = model.to(dtype=torch.float64)
                elif args.use_float32:
                    model = model.to(dtype=torch.float32)
            else:
                raise FileNotFoundError(f"Model file not found: {state_dict_path}")
        else:
            raise ValueError(f"Unsupported image model: {model_name}")
        
        model.eval()
        
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image file not found: {args.image_path}")
        
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pil_image = Image.open(args.image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image = image_transform(pil_image).unsqueeze(0)
        
        input_hash = hashlib.sha256(args.image_path.encode()).hexdigest()[:16]
    
    input_dir = os.path.join(args.record_dir, input_hash)
    
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    
    os.makedirs(input_dir, exist_ok=True)
    
    recorder = LayerRecorder(
        save_dir=input_dir,
        steps_per_block=1,
        layers_per_block=args.layers_per_block,
        checkpoint_interval=1,
        activation_interval=args.activation_interval,
        enabled=not args.disabled,
        chunk_size=chunk_size,
        use_blake3=use_blake3,
        learning_rate=None,
        module_structure_dir=args.module_structure_dir,
        model_name=model_name,
        compress_storage=False
    )
    
    start_disk_size = get_dir_size(input_dir) if recorder.enabled else 0
    
    if is_llm:
        inference_time = record_inference_llm(model, tokenizer, args.prompt, recorder, args.device)
    else:
        inference_time = record_inference_image(model, model_hooks, image, recorder, args.device, model_name)
    
    end_disk_size = get_dir_size(input_dir) if recorder.enabled else 0
    
    disk_consumed = end_disk_size - start_disk_size
    
    print("\n" + "="*80)
    print("Inference statistics")
    print("="*80)
    print(f"Inference time: {inference_time:.3f} seconds")
    if recorder.enabled:
        print(f"Disk consumption: {format_size(disk_consumed)} ({disk_consumed:,} bytes)")
    else:
        print("Recording function disabled")
    print("="*80)


if __name__ == '__main__':
    main()
