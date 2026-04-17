
import os
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[object] = None
    ):
        assert len(image_paths) == len(labels), \
            f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.num_classes = len(self.classes)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        label_counts = Counter(self.labels)
        
        distribution = {}
        for class_idx, class_name in enumerate(self.classes):
            distribution[class_name] = label_counts.get(class_idx, 0)
        
        return distribution


def load_dataset_from_directory(
    data_dir: str,
    split: str = 'train'
) -> Tuple[List[str], List[int]]:
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    image_paths = []
    labels = []
    
    split_dir = Path(data_dir) / split
    
    for class_name in classes:
        class_dir = split_dir / class_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist!")
            continue
        
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])
    
    print(f"\n[{split.upper()}] Dataset loaded:")
    print(f"  Total images: {len(image_paths)}")
    
    label_counts = Counter(labels)
    for class_name, class_idx in class_to_idx.items():
        count = label_counts.get(class_idx, 0)
        print(f"  - {class_name}: {count}")
    
    return image_paths, labels


if __name__ == '__main__':
    from src.data.augmentation import get_train_transforms, get_val_transforms
    
    train_paths, train_labels = load_dataset_from_directory(
        data_dir='data',
        split='train'
    )
    
    train_dataset = BrainTumorDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=get_train_transforms(use_augmentation=True)
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    print(f"Class distribution: {train_dataset.get_class_distribution()}")
    
    image, label = train_dataset[0]
    print(f"\nSample shape: {image.shape}")
    print(f"Sample label: {label} ({train_dataset.classes[label]})")