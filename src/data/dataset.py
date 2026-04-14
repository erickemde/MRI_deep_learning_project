"""
Brain Tumor MRI Dataset with Albumentations Support

"""

import os
import cv2
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Get class names (sorted alphabetically)
        self.classes = sorted([
            d for d in os.listdir(root) 
            if os.path.isdir(os.path.join(root, d))
        ])
        
        # Load image paths and labels
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label_idx)
        
        print(f"[Dataset] Loaded {len(self.images)} images from {root}")
        print(f"[Dataset] Classes: {self.classes}")
        for idx, class_name in enumerate(self.classes):
            count = self.labels.count(idx)
            print(f"[Dataset]   - {class_name}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        # Load image as grayscale
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        label = self.labels[idx]
        
        # Apply augmentation transform
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label