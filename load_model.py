import torch
import torchvision.models as models
from transformers import ViTForImageClassification, Dinov2ForImageClassification, AutoTokenizer, AutoModelForCausalLM

class ResNetFinalLayers(torch.nn.Module):
    def __init__(self, avgpool, fc):
        super().__init__()
        self.avgpool = avgpool
        self.fc = fc
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def load_resnet152(device, pretrained=True):
    model = models.resnet152(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, 1000)
    
    model = model.to(device)
    model = model.to(dtype=torch.bfloat16)
    
    initial_layers = torch.nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool
    )
    
    model.initial_layers = initial_layers
    
    final_layers = ResNetFinalLayers(model.avgpool, model.fc)
    model.final_layers = final_layers
        
    def new_forward(x):
        x = model.initial_layers(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.final_layers(x)
        return x
    
    model.forward = new_forward
    
    model_hooks = [
        (model.initial_layers, 'initial_layers'),
        (model.layer1, 'layer1'),
        (model.layer2, 'layer2'),
        (model.layer3, 'layer3'),
        (model.layer4, 'layer4'),
        (model.final_layers, 'final_layers')
    ]
    
    return model, model_hooks

def load_vit_large(device, pretrained=True):
    model = ViTForImageClassification.from_pretrained(
        'google/vit-large-patch16-224' if pretrained else None,
        num_labels=1000
    )
    
    model = model.to(device)
    model = model.to(dtype=torch.bfloat16)
    
    model_hooks = [
        (model.vit.embeddings, 'embeddings'),
    ]
    
    num_layers = len(model.vit.encoder.layer)
    for i in range(num_layers):
        model_hooks.append((model.vit.encoder.layer[i], f'encoder_layer_{i}'))
    
    model_hooks.extend([
        (model.vit.layernorm, 'layernorm'),
        (model.classifier, 'classifier')
    ])
    
    return model, model_hooks

def load_dinov2_giant(device, pretrained=True):
    model = Dinov2ForImageClassification.from_pretrained(
        'facebook/dinov2-giant' if pretrained else None,
        num_labels=1000
    )
    
    model = model.to(device)
    model = model.to(dtype=torch.bfloat16)
    
    model_hooks = [
        (model.dinov2.embeddings, 'embeddings'),
    ]
    
    num_layers = len(model.dinov2.encoder.layer)
    for i in range(num_layers):
        model_hooks.append((model.dinov2.encoder.layer[i], f'encoder_layer_{i}'))
    
    model_hooks.extend([
        (model.dinov2.layernorm, 'layernorm'),
        (model.classifier, 'classifier')
    ])
    
    return model, model_hooks


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
