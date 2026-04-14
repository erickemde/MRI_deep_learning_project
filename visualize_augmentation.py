"""
Augmentation Visualization 

"""

import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import BrainTumorDataset
from src.data.augmentation import (
    get_train_augmentation, 
    get_val_augmentation,
    get_light_augmentation
)


def denormalize_image(tensor_image):

    img = tensor_image.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    return img


def visualize_augmentation_comparison():
    
    print("\n" + "="*70)
    print("AUGMENTATION VISUALIZATION")
    print("="*70 + "\n")
    
    # Load datasets
    train_transform = get_train_augmentation()
    val_transform = get_val_augmentation()
    
    aug_dataset = BrainTumorDataset('data/train', transform=train_transform)
    orig_dataset = BrainTumorDataset('data/train', transform=val_transform)
    
    # Configuration
    num_samples = 5
    num_augmentations = 5
    
    # Create figure
    fig, axes = plt.subplots(
        num_samples, 
        num_augmentations + 1, 
        figsize=(18, num_samples * 3)
    )
    
    print("Generating augmentation examples...\n")
    
    for sample_idx in range(num_samples):
        # Original image
        orig_img, label = orig_dataset[sample_idx]
        axes[sample_idx, 0].imshow(denormalize_image(orig_img))
        axes[sample_idx, 0].set_title(
            f'Original\n(Class: {orig_dataset.classes[label]})', 
            fontsize=11,
            fontweight='bold'
        )
        axes[sample_idx, 0].axis('off')
        
        # Augmented versions
        for aug_idx in range(num_augmentations):
            aug_img, _ = aug_dataset[sample_idx]
            axes[sample_idx, aug_idx + 1].imshow(denormalize_image(aug_img))
            axes[sample_idx, aug_idx + 1].set_title(
                f'Augmented #{aug_idx + 1}', 
                fontsize=11
            )
            axes[sample_idx, aug_idx + 1].axis('off')
        
        print(f"   Sample {sample_idx + 1}/{num_samples} completed")
    
    # Finalize plot
    plt.suptitle(
        'Data Augmentation Examples: Original vs Augmented', 
        fontsize=16, 
        y=0.995,
        fontweight='bold'
    )
    plt.tight_layout()
    
    # Save
    output_path = 'augmentation_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n{'='*70}")
    print(f"SAVED: {output_path}")
    print(f"{'='*70}\n")
    
    plt.close()


def compare_augmentation_strengths():
    
    print("\n" + "="*70)
    print("AUGMENTATION STRENGTH COMPARISON")
    print("="*70 + "\n")
    
    transforms = {
        'No Aug': get_val_augmentation(),
        'Light Aug': get_light_augmentation(),
        'Full Aug': get_train_augmentation()
    }
    
    datasets = {
        name: BrainTumorDataset('data/train', transform=transform)
        for name, transform in transforms.items()
    }
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    sample_idx = 0
    
    for col_idx, (name, dataset) in enumerate(datasets.items()):
        for row_idx in range(3):
            img, label = dataset[sample_idx]
            axes[row_idx, col_idx].imshow(denormalize_image(img))
            
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    name, 
                    fontsize=13,
                    fontweight='bold'
                )
            axes[row_idx, col_idx].axis('off')
    
    # Original image in last column
    orig_dataset = BrainTumorDataset('data/train', transform=get_val_augmentation())
    for row_idx in range(3):
        orig_img, label = orig_dataset[sample_idx]
        axes[row_idx, 3].imshow(denormalize_image(orig_img))
        if row_idx == 0:
            axes[row_idx, 3].set_title(
                'Original', 
                fontsize=13,
                fontweight='bold'
            )
        axes[row_idx, 3].axis('off')
    
    plt.suptitle(
        'Augmentation Strength Comparison', 
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout()
    
    output_path = 'augmentation_strength_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"{'='*70}")
    print(f"SAVED: {output_path}")
    print(f"{'='*70}\n")
    
    plt.close()


if __name__ == "__main__":
    # Visualize augmentation examples
    visualize_augmentation_comparison()
    
    # Compare augmentation strengths
    compare_augmentation_strengths()
    
    print("\nVisualization complete! Check the generated PNG files.\n")