"""
Training Script with Data Augmentation
"""

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from src.models.lit_vgg import LitVGG
from src.data.augmentation import get_train_augmentation, get_val_augmentation
from src.data.dataset import BrainTumorDataset

torch.set_float32_matmul_precision('high')


def train_with_augmentation(
    experiment_name="full_augmentation",
    epochs=50,
    batch_size=64
):

    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*70 + "\n")
    
    # Initialize model
    model = LitVGG()
    
    # Get augmentation transforms
    train_transform = get_train_augmentation()
    val_transform = get_val_augmentation()
    
    # Create datasets
    print("[1/5] Loading datasets...")
    train_dataset = BrainTumorDataset(
        root="data/train",
        transform=train_transform
    )
    val_dataset = BrainTumorDataset(
        root="data/val",
        transform=val_transform
    )
    
    # Create dataloaders
    print("\n[2/5] Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup callbacks
    print("\n[3/5] Setting up callbacks...")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{experiment_name}",
        filename="best-{epoch:02d}-{valid_acc:.4f}",
        monitor="valid/valid_acc",
        mode="max",
        save_top_k=1,
        verbose=True
    )
    
    # Setup loggers
    print("\n[4/5] Setting up loggers...")
    tb_logger = TensorBoardLogger(
        save_dir="logs/",
        name=experiment_name
    )
    loggers = [tb_logger]
    
    try:
        wandb_logger = WandbLogger(
            project="brain_tumor_augmentation",
            name=experiment_name,
            log_model="all"
        )
        loggers.append(wandb_logger)
        print("   - TensorBoard: enabled")
        print("   - WandB: enabled")
    except Exception:
        print("   - TensorBoard: enabled")
        print("   - WandB: disabled")
    
    # Create trainer
    print("\n[5/5] Creating trainer...")
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        logger=loggers,
        callbacks=[lr_monitor, checkpoint_callback],
        deterministic=False,
        log_every_n_steps=10
    )
    
    # Train
    print("\nStarting training...")
    print("-"*70)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    print("\n" + "="*70)
    print(f"EXPERIMENT '{experiment_name}' COMPLETED")
    print("="*70 + "\n")
    
    return trainer


if __name__ == "__main__":
    # Run experiment with full augmentation
    train_with_augmentation(
        experiment_name="full_augmentation",
        epochs=50,
        batch_size=64
    )