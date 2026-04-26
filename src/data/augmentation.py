
import torch
from torchvision.transforms import v2


def get_train_transforms(use_augmentation=True):
    if use_augmentation:
        return v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),

            v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomRotation(
            #     degrees=10,  # 15 → 10
            #     interpolation=v2.InterpolationMode.BILINEAR
            # ),
            
            v2.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=v2.InterpolationMode.BILINEAR
            ),
            
            # MRI pixel intensities carry direct diagnostic meaning — specific signal windows correspond to tissue types 
            # (white matter, tumor mass, edema). ColorJitter randomly shifts brightness/contrast in [0,1] space, 
            # corrupting those clinically meaningful distributions. Natural-image augmentation logic doesn't transfer to 
            # grayscale medical images.

            # v2.ColorJitter(
            #     brightness=0.1,  # 0.2 → 0.1
            #     contrast=0.1     # 0.2 → 0.1
            # ),
            
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            # ,
            # v2.GaussianNoise(   #mimics MRI acquisition noise and is more medically grounded than ColorJitter.
            #     mean=0.0, 
            #     sigma=0.02, 
            #     clip=True
            # )
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