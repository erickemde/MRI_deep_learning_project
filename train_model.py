import argparse
import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset, load_dataset_from_directory
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.visualization.gradcam import GradCAM
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from src.experiments.config import setup_experiment, build_model
from huggingface_upload import huggingface_upload_model, check_hf_login
import yaml
from pathlib import Path
from src.models.LightningWrapper import LightningWrapper

torch.set_float32_matmul_precision('medium')


def main():
    hf_status = check_hf_login()

    parser = argparse.ArgumentParser(description='Brain Tumor Classification')
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = setup_experiment(config)

    pl.seed_everything(config['seed'], workers=True)

    use_augmentation = config['use_augmentation']
    model_type = config['model_description']
    experiment_name = config['experiment_name']

    print("=" * 70)
    print(f"EXPERIMENT: {config['experiment']}")
    print("=" * 70)
    print(f"  Model Type: {model_type}")
    print(f"  Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
    print(f"  Training Mode: Feature Extraction (Frozen Backbone)")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  Seed: {config['seed']}")
    print("=" * 70)

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

    if use_augmentation:
        print("  Augmentation enabled:")
        print("     - RandomHorizontalFlip")
        print("     - RandomAffine")
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
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        worker_init_fn=lambda _: None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    print("\n[4/5] Creating model...")

    model = build_model(config)

    print("\n[5/5] Setting up trainer...")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(config['checkpoint_dir']),
        filename='best-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    tb_logger = TensorBoardLogger(save_dir="logs/", name=experiment_name)
    loggers = [tb_logger]
    try:
        wandb_logger = WandbLogger(project="deep_learning_project", log_model="all")
        loggers.append(wandb_logger)
    except Exception:
        print("Wandb not available, logging to Tensorboard only.")

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, lr_callback],
        logger=loggers,
        enable_progress_bar=True,
        deterministic="warn",
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

    if config.get('generate_gradcam', True):
        print("\n" + "=" * 70)
        print("GENERATING GRADCAM VISUALIZATIONS")
        print("=" * 70)

        try:
            checkpoint_stem = Path(checkpoint_callback.best_model_path).stem
            gradcam_save_dir = os.path.join("gradcam_examples", experiment_name, checkpoint_stem)
            model = LightningWrapper.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                model = build_model(config).model
            )
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            gradcam = GradCAM(model, model.gradcam_target_layer)
            gradcam.examples(
                data_subset="val",
                save_dir=gradcam_save_dir,
                total_examples=config["total_examples"],
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                seed=config["seed"]
            )
        except Exception as e:
            print(f"[WARNING] GradCAM visualization failed: {e}")

    if hf_status:
        print("\n" + "=" * 70)
        print("SAVING TO HUGGINGFACE")
        print("=" * 70)

        huggingface_upload_model(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    main()