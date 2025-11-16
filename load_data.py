import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms


class ChatDataset(Dataset):
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        ds = load_dataset("ShenLab/MentalChat16K")
        
        self.samples = []
        for split in ['train', 'validation']:
            if split in ds:
                for item in ds[split]:
                    user_content = item.get('input')
                    assistant_content = item.get('output')
                    if user_content is not None and assistant_content is not None:
                        self.samples.append([
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        messages = self.samples[idx]
        
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        encoded = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        
        if input_ids.max() >= self.tokenizer.vocab_size:
            input_ids = torch.clamp(input_ids, max=self.tokenizer.vocab_size - 1)
        
        return input_ids

def get_llm_dataloader(tokenizer, batch_size=4, max_length=128, seed=None):
    dataset = ChatDataset(tokenizer, max_length)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


class ImageNetDataset(Dataset):
    def __init__(self, split='train', image_size=224, data_offset=0):
        self.split = split
        self.image_size = image_size
        self.data_offset = data_offset
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        ds = load_dataset("timm/mini-imagenet", split=split)
        
        self.samples = []
        for item in ds:
            image = item['image']
            label = item['label']
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.samples.append((image, label))
        
        if data_offset > 0:
            self.samples = self.samples[data_offset:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = self.transform(image)
        return image, label

def get_imagenet_dataloader(batch_size=32, image_size=224, split='train', num_workers=4, data_offset=0, seed=None, shuffle=True):
    dataset = ImageNetDataset(split=split, image_size=image_size, data_offset=data_offset)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator)
