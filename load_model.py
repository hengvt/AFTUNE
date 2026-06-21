import os
import torch
from transformers import ViTForImageClassification, Dinov2ForImageClassification, AutoTokenizer, AutoModelForCausalLM

def load_vit_large(device, pretrained=True):
    model = ViTForImageClassification.from_pretrained(
        'google/vit-large-patch16-224' if pretrained else None,
        num_labels=1000
    )
    
    model = model.to(device)
    model = model.to(dtype=torch.bfloat16)
    
    layer_tracked = [
        (model.vit.embeddings, 'embeddings'),
    ]
    
    num_layers = len(model.vit.encoder.layer)
    for i in range(num_layers):
        layer_tracked.append((model.vit.encoder.layer[i], f'encoder_layer_{i}'))
    
    layer_tracked.extend([
        (model.vit.layernorm, 'layernorm'),
        (model.classifier, 'classifier')
    ])
    
    return model, layer_tracked

def load_dinov2_giant(device, pretrained=True):
    model = Dinov2ForImageClassification.from_pretrained(
        'facebook/dinov2-giant' if pretrained else None,
        num_labels=1000
    )
    
    model = model.to(device)
    model = model.to(dtype=torch.bfloat16)
    
    layer_tracked = [
        (model.dinov2.embeddings, 'embeddings'),
    ]
    
    num_layers = len(model.dinov2.encoder.layer)
    for i in range(num_layers):
        layer_tracked.append((model.dinov2.encoder.layer[i], f'encoder_layer_{i}'))
    
    layer_tracked.extend([
        (model.dinov2.layernorm, 'layernorm'),
        (model.classifier, 'classifier')
    ])
    
    return model, layer_tracked


def load_finetuned_vision_model(model_path, model_name, device):
    if model_name == 'vit_large':
        model, layer_tracked = load_vit_large(device=device)
    elif model_name == 'dinov2_giant':
        model, layer_tracked = load_dinov2_giant(device=device)
    else:
        raise ValueError(f"Unsupported vision model: {model_name}")

    state_dict_path = os.path.join(model_path, f'{model_name}.pth')
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Model file not found: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    return model, layer_tracked


def load_image(image_path: str, image_size: int = 224):
    from PIL import Image
    from torchvision import transforms
    
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.open(image_path)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    image = image_transform(pil_image).unsqueeze(0)
    return image


def load_llm_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    
    return model, tokenizer
