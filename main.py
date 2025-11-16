import torch
import argparse
import os
import shutil
from finetune import finetune_llm_full, finetune_llm_lora, get_tracked_modules
from load_data import get_llm_dataloader, get_imagenet_dataloader
from load_model import load_resnet152, load_vit_large, load_dinov2_giant, load_llm_model
from finetune import finetune_imagenet
from layer_recorder import LayerRecorder, set_deterministic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model_type', type=str, default='llm', choices=['llm', 'resnet152', 'vit_large', 'dinov2_giant'], help='Model type')
    parser.add_argument('-p', '--model_path', type=str, default='../models/Llama-3.1-8B-Instruct', help='LLM model path')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-s', '--steps_per_block', type=int, default=1, help='Steps per checkpoint (B_S)')
    parser.add_argument('-l', '--layers_per_block', type=int, default=1, help='Layers per block (B_L)')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint save interval (C_I)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length for LLM')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--chunk_size', type=int, default=4096, help='Hash chunk size')
    parser.add_argument('--sha256', action='store_true', help='Use SHA256 instead of BLAKE3')
    parser.add_argument('--use_adam', action='store_true', help='Use AdamW optimizer')
    parser.add_argument('--disabled', action='store_true', help='Disable AFTUNE')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--target_modules', type=str, nargs='+', default="['q_proj', 'v_proj']")
    parser.add_argument('--module_structure_dir', type=str, default='./module_structures')
    parser.add_argument('--record_dir', type=str, default='./block_records')
    parser.add_argument('--output_dir', type=str, default='./finetuned_model')
    parser.add_argument('--warmup_steps', type=int, default=0)
    args = parser.parse_args()
    
    set_deterministic(seed=args.seed, enabled=args.deterministic)
    
    if os.path.exists(args.record_dir):
        shutil.rmtree(args.record_dir)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    model_name_map = {
        'resnet152': 'resnet152',
        'vit_large': 'vit_large',
        'dinov2_giant': 'dinov2_giant'
    }
    model_name = model_name_map.get(args.model_type)
    if args.model_type == 'llm' and args.model_path:
        model_name = os.path.basename(os.path.normpath(args.model_path))
    
    recorder = LayerRecorder(
        save_dir=args.record_dir,
        steps_per_block=args.steps_per_block,
        layers_per_block=args.layers_per_block,
        checkpoint_interval=args.checkpoint_interval,
        enabled=not args.disabled,
        chunk_size=args.chunk_size,
        use_blake3=not args.sha256,
        learning_rate=args.lr,
        optimizer_type='adamw' if args.use_adam else 'sgd',
        module_structure_dir=args.module_structure_dir,
        model_name=model_name,
    )
    
    tracked_modules = None
    
    if args.model_type == 'llm':
        model, tokenizer = load_llm_model(args.model_path, args.device)
        dataloader = get_llm_dataloader(tokenizer, args.batch_size, args.max_length)
        
        if args.use_lora:
            finetune_llm_lora(
                model, dataloader, recorder, 
                epochs=args.epochs, 
                learning_rate=args.lr, 
                max_samples=args.max_samples,
                target_modules=args.target_modules,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                warmup_steps=args.warmup_steps
            )
        else:
            finetune_llm_full(model, dataloader, recorder, args.epochs, args.lr, max_samples=args.max_samples, warmup_steps=args.warmup_steps)
        
        tracked_modules = get_tracked_modules(model)
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    elif args.model_type == 'resnet152':
        model, model_hooks = load_resnet152(device=args.device)
        dataloader = get_imagenet_dataloader(batch_size=args.batch_size)
        
        finetune_imagenet(model, dataloader, recorder, model_hooks, 
                        epochs=args.epochs, learning_rate=args.lr, max_samples=args.max_samples if args.max_samples is not None else 8192,
                        warmup_steps=args.warmup_steps)
        
        tracked_modules = {name: module for module, name in model_hooks}
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'resnet152.pth'))
    
    elif args.model_type == 'vit_large':
        model, model_hooks = load_vit_large(device=args.device)
        dataloader = get_imagenet_dataloader(batch_size=args.batch_size)
        
        finetune_imagenet(model, dataloader, recorder, model_hooks, 
                        epochs=args.epochs, learning_rate=args.lr, max_samples=args.max_samples if args.max_samples is not None else 8192,
                        warmup_steps=args.warmup_steps)
        
        tracked_modules = {name: module for module, name in model_hooks}
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
    
    elif args.model_type == 'dinov2_giant':
        model, model_hooks = load_dinov2_giant(device=args.device)
        dataloader = get_imagenet_dataloader(batch_size=args.batch_size)
        
        finetune_imagenet(model, dataloader, recorder, model_hooks, 
                        epochs=args.epochs, learning_rate=args.lr, max_samples=args.max_samples if args.max_samples is not None else 8192,
                        warmup_steps=args.warmup_steps)
        
        tracked_modules = {name: module for module, name in model_hooks}
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'dinov2_giant.pth'))
    
    recorder.save_all_records()
    
    if recorder.enabled and tracked_modules is not None:
        recorder.save_final_model_parameters(model, tracked_modules)
    
    recorder.print_time_stats()
    recorder.print_storage_stats()
