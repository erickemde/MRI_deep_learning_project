import argparse
import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset, load_dataset_from_directory
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.models.lit_vgg_attention import VGG19Baseline, VGG19SEAttention, VGG19SoftmaxAttention

torch.set_float32_matmul_precision('medium')


def main():
    parser = argparse.ArgumentParser(description='Brain Tumor Classification')
    
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['baseline', 'baseline_aug', 
                               'softmax_attention', 'softmax_attention_finetune',
                               'se_attention', 'se_attention_aug'],
                       help='Experiment type')
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"EXPERIMENT: {args.experiment}")
    print("=" * 70)
    
    if args.experiment == 'baseline':
        use_augmentation = False
        model_type = 'baseline'
        experiment_name = 'vgg19_baseline'
        
    elif args.experiment == 'baseline_aug':
        use_augmentation = True
        model_type = 'baseline'
        experiment_name = 'vgg19_baseline_aug'
        
    elif args.experiment == 'softmax_attention':
        use_augmentation = False
        model_type = 'softmax_attention'
        experiment_name = 'vgg19_softmax_attention'
        
    elif args.experiment == 'softmax_attention_finetune':
        use_augmentation = False
        model_type = 'softmax_attention_finetune'
        experiment_name = 'vgg19_softmax_attention_finetune'
        
    elif args.experiment == 'se_attention':
        use_augmentation = False
        model_type = 'se_attention'
        experiment_name = 'vgg19_se_attention'
        
    elif args.experiment == 'se_attention_aug':
        use_augmentation = True
        model_type = 'se_attention'
        experiment_name = 'vgg19_se_attention_aug'
    
    print(f"  Model Type: {model_type}")
    print(f"  Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
    print(f"  Training Mode: Feature Extraction (Frozen Backbone)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print("=" * 70)
    
    print("\n[1/5] Loading datasets...")
    
    train_paths, train_labels = load_dataset_from_directory(
        data_dir=args.data_dir,
        split='train'
    )
    
    val_paths, val_labels = load_dataset_from_directory(
        data_dir=args.data_dir,
        split='val'
    )
    
    print("\n[2/5] Creating transforms...")
    
    if use_augmentation:
        print("  Augmentation enabled:")
        print("     - RandomHorizontalFlip")
        print("     - RandomRotation")
        print("     - RandomAffine")
        print("     - ColorJitter")
    else:
        print("  No augmentation")
    
    train_transform = get_train_transforms(use_augmentation=use_augmentation)
    val_transform = get_val_transforms()
    
    print("\n[3/5] Creating datasets...")
    
    train_dataset = BrainTumorDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=train_transform
    )
    
    val_dataset = BrainTumorDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    print("\n[4/5] Creating model...")
    
    if model_type == 'baseline':
        model = VGG19Baseline(
            num_classes=4,
            lr=args.lr,
            pretrained=True
        )
        print("  Model: VGG19 Baseline")
        
    elif model_type == 'softmax_attention':
        model = VGG19SoftmaxAttention(
            num_classes=4,
            lr=args.lr,
            pretrained=True,
            unfreeze_from_layer=None
        )
        print("  Model: VGG19 + Softmax Attention (Frozen)")
        
    elif model_type == 'softmax_attention_finetune':
        model = VGG19SoftmaxAttention(
            num_classes=4,
            lr=args.lr,
            pretrained=True,
            unfreeze_from_layer=35
        )
        print("  Model: VGG19 + Softmax Attention (Partial Fine-tune)")
        
    elif model_type == 'se_attention':
        model = VGG19SEAttention(
            num_classes=4,
            reduction=16,
            lr=args.lr,
            pretrained=True
        )
        print("  Model: VGG19 + SE Attention")
    
    print("\n[5/5] Setting up trainer...")
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, experiment_name),
        filename='best-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
        log_every_n_steps=10
    )
    
    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70 + "\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT '{experiment_name}' COMPLETED")
    print(f"  Best Validation Accuracy: {checkpoint_callback.best_model_score:.4f}")
    print(f"  Checkpoint: {checkpoint_callback.best_model_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()