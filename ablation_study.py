import argparse
import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset, load_dataset_from_directory
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.models.lit_vgg_attention import VGGLightningWrapper, VGG19Baseline, VGG19SEAttention, VGG19SoftmaxAttention
from src.models.lit_vgg import LitVGG
from src.visualization.gradcam import GradCAM
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.experiments.config import setup_experiment, build_model, setup_ablation_study
import yaml
from pathlib import Path

torch.set_float32_matmul_precision('medium')

def _parse_config():
    '''
    Parse config and return the configuration of an experiment or ablation study
    '''
    parser = argparse.ArgumentParser(description='Brain Tumor Classification')    
    
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print(f"EXPERIMENT: {config['experiment']}")
    print("=" * 70)

    if config['experiment'] == "ablation":
        # Create dict of configs for each experiment
        experiment_dict = setup_ablation_study(config)

    else:
        # Create dict of single experiment
        config = setup_experiment(config)
        experiment_dict = {config['experiment']: config}

    return experiment_dict, config

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

    print("\n[2/5] Creating transforms...")

    if augment:
        print("  Augmentation enabled:")
        print("     - RandomHorizontalFlip")
        print("     - RandomRotation")
        print("     - RandomAffine")
        print("     - ColorJitter")
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

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader, train_aug_loader

def _logger_setup(config):
    # create loggers
    tb_logger = TensorBoardLogger(save_dir="logs/", name=config['experiment_name'])
    loggers = [tb_logger]
    try:
        wandb_logger = WandbLogger(project = "deep_learning_project", log_model="all", name=config['experiment_name'])
        loggers.append(wandb_logger)
    except Exception:
        print("Wandb not available, logging to Tensorboard only.")
    return loggers

def _trainer_setup(config):
    '''
    Set up checkpoints, callbacks, loggers, and pytorch_lightning trainer
    '''
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(config["checkpoint_dir"]),
        filename='best-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    loggers = _logger_setup(config)

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, lr_callback],
        logger=loggers,
        enable_progress_bar=False,
    )

    return trainer, checkpoint_callback

def main():
    # Parse config
    experiment_dict, config = _parse_config()

    # Load data
    get_augmented_data = any([v['use_augmentation'] for v in experiment_dict.values()]) # load augmented data if any experiments use augmentation
    train_loader, val_loader, train_aug_loader = _load_augment_data(config, get_augmented_data)

    # Build and train models
    print("\n[4/5] Creating model(s)...")
    models = {}
    n_models = len(experiment_dict.keys())
    for i, (name, c) in enumerate(experiment_dict.items()):
        model = build_model(c)

        # Set up trainer and loggers
        print("\n[5/5] Setting up trainer...")
        trainer, checkpoint_callback = _trainer_setup(c)

        print("\n" + "=" * 70)
        print(f"TRAINING STARTED: Experiment {name} [{i+1}/{n_models}]")
        print("=" * 70 + "\n")
    
        print(f"  Model Type: {c['model_description']}")
        print(f"  Augmentation: {'Enabled' if c['use_augmentation'] else 'Disabled'}")
        print(f"  Training Mode: Feature Extraction (Frozen Backbone)")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Learning Rate: {config['lr']}")
        print("=" * 70)

        # Train model
        if c['use_augmentation']:
            trainer.fit(model, train_aug_loader, val_loader)
        else:
            trainer.fit(model, train_loader, val_loader)

        print("\n" + "=" * 70)
        print(f"EXPERIMENT '{c['experiment_name']}' COMPLETED")
        print(f"  Best Validation Accuracy: {checkpoint_callback.best_model_score:.4f}")
        print(f"  Checkpoint: {checkpoint_callback.best_model_path}")
        print("=" * 70)

        if config['generate_gradcam']:
            # Generate GradCAM visualizations
            print("\n" + "=" * 70)
            print("GENERATING GRADCAM VISUALIZATIONS")
            print("=" * 70)
            
            try:
                checkpoint_stem = Path(checkpoint_callback.best_model_path).stem
                gradcam_save_dir = os.path.join("gradcam_examples", c['experiment_name'], checkpoint_stem)
                model = model.to("cuda" if torch.cuda.is_available() else "cpu")
                gradcam = GradCAM(model, model.gradcam_target_layer)
                gradcam.examples(
                    dataloader=val_loader,
                    save_dir=gradcam_save_dir,
                    total_examples=config["total_examples"],
                    seed=config["seed"]
                )
            except Exception as e:
                print(f"[WARNING] GradCAM visualization failed: {e}")

        models[name] = model

if __name__ == '__main__':
    main()