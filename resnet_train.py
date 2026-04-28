from typing import Any
import wandb
from src.data.dataset import BrainTumorDataset, load_dataset_from_directory
from src.data.augmentation import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as t_models
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning as pl
import os
import sys
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.models.LightningWrapper import LightningWrapper
from huggingface_upload import huggingface_upload_model, check_hf_login

torch.set_float32_matmul_precision('medium')

CONFIG = {
    'train_model': True,
    'data_dir': 'data',
    'batch_size': 64,
    'num_workers': 1,
    'checkpoint_dir': 'checkpoint',
    'experiment_name': 'resnet_aug_cbam4_conv',
    'epochs': 10,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'use_augmentation': True, # True | False
    'tune_layer4': False,
    'cbam_layer3': False, # True | False : add CBAM after layer3?
    'cbam_layer4': True, # True | False : add CBAM after layer4?
    'attention': 'conv', # conv | self : spatial attention type
    'hf_upload': True,
}

# Load train and valid data
def _load_augment_data(config, augment):
    print("\n[1/5] Loading datasets...")

    train_paths, train_labels = load_dataset_from_directory(
        data_dir=config['data_dir'],
        split='train'
    )
    
    val_paths, val_labels = load_dataset_from_directory(
        data_dir=config['data_dir'],
        split='val'
    )

    test_paths, test_labels = load_dataset_from_directory(
        data_dir=config['data_dir'],
        split='test'
    )

    print("\n[2/5] Creating transforms...")

    if augment:
        print("  Augmentation enabled:")
        print("     - RandomHorizontalFlip")
        print("     - RandomAffine")
    else:
        print("  No augmentation")

    # Get regular dataset
    train_transform = get_train_transforms(use_augmentation=False)
    val_transform = get_val_transforms()

    # Get augmented dataset
    if augment:
        train_augmented = get_train_transforms(use_augmentation=True)

    print("\n[3/5] Creating datasets...")
    
    train_dataset = BrainTumorDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=train_transform
    )

    train_aug_loader = None
    if augment:
        train_aug_dataset = BrainTumorDataset(
            image_paths=train_paths,
            labels=train_labels,
            transform=train_augmented
        )

        train_aug_loader = DataLoader(
            train_aug_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True
        )        
    
    val_dataset = BrainTumorDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=val_transform
    )
    
    test_dataset = BrainTumorDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, train_aug_loader, test_loader

# Build model
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1) # reduce image size to 1x1, max across all channels
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1) # reduce image size to 1x1, avg across all channels
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels//reduction),
            nn.ReLU(),
            nn.Linear(in_features=in_channels//reduction, out_features=in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        N,C,_,_ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(N,C))
        max_out = self.mlp(self.max_pool(x).view(N,C))
        a = self.sigmoid((avg_out+max_out).view(N,C,1,1))
        return a * x

class SpatialConvAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        a = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return a * x
    
class SpatialSelfAttention(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.q = nn.Linear(in_features=2, out_features=embed_dim)
        self.k = nn.Linear(in_features=2, out_features=embed_dim)
        self.v = nn.Linear(in_features=2, out_features=embed_dim)
        self.proj = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        N, C, h, w = x.shape

        avg_out = torch.mean(x, dim=1, keepdim=True) # N, 1, h, w
        max_out = torch.amax(x, dim=1, keepdim=True) # N, 1, h, w
        tokens = torch.cat([avg_out, max_out], dim=1).flatten(start_dim=2).permute(0,2,1) # N, h*w, 2

        q = self.q(tokens) # query; N, h*w, embed_dim
        k = self.k(tokens) # key
        v = self.v(tokens) # value

        out, out_weights = self.attn(query=q, key=k, value=v)
        out_proj = self.proj(out) # N, h*w, 1
        out_proj = out_proj.permute(0,2,1).view(N, 1, h, w) # N, 1, h, w
        a = self.sigmoid(out_proj)
        return a * x
    
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, attention='conv', kernel_size=7, embed_dim=16, num_heads=2):
        super().__init__()
        assert attention in ['conv', 'self'], f"CBAM attention: {attention} is invalid."
        self.channel_attn = ChannelAttention(in_channels=in_channels, reduction=reduction)
        if attention=='conv':
            self.spatial_attn = SpatialConvAttention(kernel_size=kernel_size)
        elif attention=='self':
            self.spatial_attn = SpatialSelfAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        a = self.channel_attn(x)
        a = self.spatial_attn(a)
        return x + a # residual skip connection
    
def build_model(tune_layer4 = False, cbam_layer3 = False, cbam_layer4 = False, attention = 'conv', num_classes=4):
    # Init model
    model = t_models.resnet34(weights=t_models.ResNet34_Weights.DEFAULT)
    
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # Insert CBAM
    if cbam_layer3:
        model.layer4 = nn.Sequential(
            CBAM(in_channels=256, attention=attention), # this is unfrozen
            model.layer4
        )
        print(f"Inserted CBAM {attention}-attention before layer4...")

    if cbam_layer4:
        model.layer4 = nn.Sequential(
            model.layer4, # this stays frozen
            CBAM(in_channels=512, attention=attention) # this is unfrozen
        )
        print(f"Inserted CBAM {attention}-attention after layer4...")

    print(f"Tune layer4? {tune_layer4}.")
    if tune_layer4:
        # unfreeze layer 4
        print("Unfreezing layer4...")
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Add classifier
    model.fc = nn.Linear(in_features=512, out_features=num_classes)

    model.num_classes = num_classes

    # View model architecture
    print(model)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    layer3_trainable = sum(p.numel() for p in model.layer3.parameters() if p.requires_grad)
    layer4_trainable = sum(p.numel() for p in model.layer4.parameters() if p.requires_grad)
    layer4_total = sum(p.numel() for p in model.layer4.parameters())
    print(f"Total params: {total_params:,}; trainable params: {trainable_params:,}; ratio: {trainable_params/total_params:.4f}")
    print(f"- layer3 trainable params: {layer3_trainable:,}")
    print(f"- layer4 total params: {layer4_total:,}; trainable params: {layer4_trainable:,}; ratio: {layer4_trainable/layer4_total:.4f}")

    return model

def _logger_setup(config):
    # create loggers
    tb_logger = TensorBoardLogger(save_dir="logs/", name=config['experiment_name'])
    try:
        wandb_logger = WandbLogger(project = "deep_learning_project", log_model="all", name=config['experiment_name'])
    except Exception:
        wandb_logger = None
        print("Wandb not available, logging to Tensorboard only.")
    return tb_logger, wandb_logger

def _trainer_setup(config):
    '''
    Set up checkpoints, callbacks, loggers, and pytorch_lightning trainer
    '''
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["checkpoint_dir"]),
        filename='best-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    tb_logger, wandb_logger = _logger_setup(config)

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, lr_callback],
        logger=[tb_logger, wandb_logger],
        enable_progress_bar=True,
    )

    return trainer, checkpoint_callback

# Train model
def main():
    wandb.finish()
    pl.seed_everything(42, workers=True)

    # Load data
    train_loader, val_loader, train_aug_loader, test_loader = _load_augment_data(CONFIG, augment=CONFIG['use_augmentation'])

    # Build model
    print("\n[4/5] Creating model...")
    model = build_model(tune_layer4=CONFIG['tune_layer4'], cbam_layer3=CONFIG['cbam_layer3'], cbam_layer4=CONFIG['cbam_layer4'], attention=CONFIG['attention'])
    lightning_model = LightningWrapper(model, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

    # Continue with training?
    if not CONFIG['train_model']:
        print(f"'train_model' = {CONFIG['train_model']}. Exiting...")
        sys.exit(0)

    # Set up trainer and loggers
    print("\n[5/5] Setting up trainer...")
    trainer, checkpoint_callback = _trainer_setup(CONFIG)

    print("\n" + "=" * 70)
    print(f"TRAINING STARTED: Experiment {CONFIG['experiment_name']}")
    print("=" * 70 + "\n")

    # Train model
    if CONFIG['use_augmentation']:
        trainer.fit(lightning_model, train_aug_loader, val_loader)
    else:
        trainer.fit(lightning_model, train_loader, val_loader)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT '{CONFIG['experiment_name']}' COMPLETED")
    print(f"  Best Validation Accuracy: {checkpoint_callback.best_model_score:.4f}")
    print(f"  Checkpoint: {checkpoint_callback.best_model_path}")
    print("=" * 70)

    wandb.finish()

    hf_status = check_hf_login()
    if hf_status and CONFIG['hf_upload']:
        print("\n" + "=" * 70)
        print("SAVING TO HUGGINGFACE")
        print("=" * 70)

        huggingface_upload_model(checkpoint_callback.best_model_path)

if __name__ == '__main__':
    main()