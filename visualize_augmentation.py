import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
import numpy as np

from src.data.dataset import load_dataset_from_directory


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def get_augmentation_variants():
    
    no_aug = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    light_aug = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=5, interpolation=v2.InterpolationMode.BILINEAR),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_aug = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), interpolation=v2.InterpolationMode.BILINEAR ),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'No Aug': no_aug,
        'Light Aug': light_aug,
        'Full Aug': full_aug
    }


def visualize_augmentation_examples(data_dir='data', num_samples=5, save_path='augmentation_examples.png'):
    
    train_paths, train_labels = load_dataset_from_directory(data_dir, split='train')
    
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    selected_images = []
    for class_idx in range(4):
        indices = [i for i, label in enumerate(train_labels) if label == class_idx]
        if indices:
            selected_images.append((train_paths[indices[0]], class_idx))
    
    if len(train_labels) > 0:
        selected_images.append((train_paths[0], train_labels[0]))
    
    aug_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), interpolation=v2.InterpolationMode.BILINEAR ),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    no_aug_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    fig, axes = plt.subplots(len(selected_images), num_samples + 1, figsize=(18, 3 * len(selected_images)))
    fig.suptitle('Data Augmentation Examples: Original vs Augmented', fontsize=16, fontweight='bold')
    
    for row, (img_path, class_idx) in enumerate(selected_images):
        img = Image.open(img_path).convert('RGB')
        
        orig_tensor = no_aug_transform(img)
        orig_img = denormalize(orig_tensor).permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
        axes[row, 0].imshow(orig_img, cmap='gray')
        axes[row, 0].set_title(f'Original\n(Class: {classes[class_idx]})', fontsize=10)
        axes[row, 0].axis('off')
        
        for col in range(1, num_samples + 1):
            aug_tensor = aug_transform(img)
            aug_img = denormalize(aug_tensor).permute(1, 2, 0).numpy()
            aug_img = np.clip(aug_img, 0, 1)
            
            axes[row, col].imshow(aug_img, cmap='gray')
            axes[row, col].set_title(f'Augmented #{col}', fontsize=10)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_augmentation_strength(data_dir='data', save_path='augmentation_strength_comparison.png'):
    
    train_paths, train_labels = load_dataset_from_directory(data_dir, split='train')
    
    import random
    random.seed(42)
    indices = random.sample(range(len(train_paths)), 3)
    
    variants = get_augmentation_variants()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Augmentation Strength Comparison', fontsize=16, fontweight='bold')
    
    titles = ['No Aug', 'Light Aug', 'Full Aug', 'Original']
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    for row, idx in enumerate(indices):
        img_path = train_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        orig_transform = v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        orig_tensor = orig_transform(img)
        orig_img = orig_tensor.permute(1, 2, 0).numpy()
        
        for col, (name, transform) in enumerate(variants.items()):
            aug_tensor = transform(img)
            aug_img = denormalize(aug_tensor).permute(1, 2, 0).numpy()
            aug_img = np.clip(aug_img, 0, 1)
            
            axes[row, col].imshow(aug_img, cmap='gray')
            axes[row, col].axis('off')
        
        axes[row, 3].imshow(orig_img, cmap='gray')
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    visualize_augmentation_examples(
        data_dir='data',
        num_samples=5,
        save_path='augmentation_examples.png'
    )
    
    visualize_augmentation_strength(
        data_dir='data',
        save_path='augmentation_strength_comparison.png'
    )