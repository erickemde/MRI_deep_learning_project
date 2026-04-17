"""
Data Augmentation for Brain Tumor MRI Images

"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation():

    return A.Compose([
        # Resize to VGG input size
        A.Resize(224, 224),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        
        A.Rotate(
            limit=15,       
            p=0.5
        ),
        
        A.Affine(
            scale=(0.9, 1.1),           
            translate_percent={
                'x': (-0.1, 0.1),       
                'y': (-0.1, 0.1)      
            },
            p=0.5
        ),
        
        # Intensity transformations (mild for MRI)
        A.RandomBrightnessContrast(
            brightness_limit=0.1,      
            contrast_limit=0.1,       
            p=0.3
        ),
        
        A.RandomGamma(
            gamma_limit=(90, 110),    
            p=0.3
        ),
        
        # Noise simulation 
        A.GaussNoise(
            var_limit=(1, 10),          
            mean=0,                   
            per_channel=False,         
            p=0.2                       
        ),
        
        # Elastic deformation

        A.ElasticTransform(
            alpha=30,                  
            sigma=5,                 
            p=0.1                    
        ),
        
        # Convert grayscale to RGB (VGG requires 3 channels)
        A.ToRGB(),
        
        # Normalize using ImageNet statistics
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2(),
    ])


def get_val_augmentation():

    return A.Compose([
        A.Resize(224, 224),
        A.ToRGB(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_light_augmentation():

    return A.Compose([
        A.Resize(224, 224),
        
        # Only basic geometric transforms
        A.HorizontalFlip(p=0.5),
        
        A.Rotate(
            limit=10,        
            p=0.3
        ),
        
        # Minimal intensity adjustment
        A.RandomBrightnessContrast(
            brightness_limit=0.05,      
            contrast_limit=0.05,
            p=0.2
        ),
        
        # NO noise in light version
        # NO elastic deformation in light version
        
        A.ToRGB(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])