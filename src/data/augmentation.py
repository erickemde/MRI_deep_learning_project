# src/data/augmentation.py

import torch
from torchvision.transforms import v2


def get_train_transforms(use_augmentation=True):
    if use_augmentation:
        return v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(
                degrees=10,  # 15 → 10
                interpolation=v2.InterpolationMode.BILINEAR
            ),
            
            v2.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=v2.InterpolationMode.BILINEAR
            ),
            
            v2.ColorJitter(
                brightness=0.1,  # 0.2 → 0.1
                contrast=0.1     # 0.2 → 0.1
            ),
            
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_val_transforms():
    return v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])