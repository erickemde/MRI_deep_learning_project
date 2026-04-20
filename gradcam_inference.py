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
from src.experiments.config import setup_experiment, build_model
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        gradcam_config = yaml.safe_load(f)
    
    with open(gradcam_config["base"]) as f:
        config = yaml.safe_load(f)

    config.update(gradcam_config)
    config = setup_experiment(config)
    experiment_name = config["experiment_name"]
    model_type = config["model_description"]
    experiment_name = config["experiment_name"]

    
    print("=" * 70)
    print(f"GRADCAM EXPERIMENT: {config["experiment"]}")
    print("=" * 70)
    
    print(f"  Model Type: {model_type}")
    print(f"  Batch Size: {config['batch_size']}")
    print("=" * 70)
    
        
    print("\n[1/4] Loading datasets...")
    
    train_paths, train_labels = load_dataset_from_directory(
        data_dir=config["data_dir"],
        split='train'
    )
    
    val_paths, val_labels = load_dataset_from_directory(
        data_dir=config["data_dir"],
        split='val'
    )
    
    print("\n[2/4] Creating datasets...")
    
    val_transform = get_val_transforms()

       
    val_dataset = BrainTumorDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    print(f"  Val batches: {len(val_loader)}")
    
    
    print("\n[3/4] Creating model...")
    model = VGGLightningWrapper.load_from_checkpoint(
        config["checkpoint_path"],
        model = build_model(config).model
    )
      
    # Generate GradCAM visualizations
    print("\n" + "=" * 70)
    print("[4/4] Generating Gradcam Visualizations")
    print("=" * 70)
    try:
        gradcam_save_dir = os.path.join("gradcam_examples", experiment_name)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        gradcam = GradCAM(model, model.gradcam_target_layer)
        gradcam.examples(
            dataloader=val_loader,
            save_dir=gradcam_save_dir,
            total_examples=3,
            seed=42
        )
        print(f"\nGradCAM visualizations saved to: {gradcam_save_dir}")
    except Exception as e:
        print(f"[WARNING] GradCAM visualization failed: {e}")
        
if __name__ == '__main__':
    main()