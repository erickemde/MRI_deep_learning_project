import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset, load_dataset_from_directory
from src.data.augmentation import get_val_transforms
import yaml
from src.models.LightningWrapper import LightningWrapper
from src.models.vgg19 import VGG19Baseline, VGG19SEAttention, VGG19SoftmaxAttention, VGG19CBAMAttention, VGG19SelfAttention

torch.set_float32_matmul_precision('medium')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _parse_config():
    parser = argparse.ArgumentParser(description='Brain Tumor Classification')
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault('experiment', 'eval')
    config.setdefault('models', {})
    print("=" * 70)
    print(f"Config experiment: {config['experiment']}")
    print("=" * 70)
    return config

def _load_val_data(config):
    print("Loading val data...")
    val_paths, val_labels = load_dataset_from_directory(
        data_dir=config['data_dir'],
        split='val'
    )
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
    return val_loader

def _evaluate(model, loader):
    model.eval()
    model.to(DEVICE)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total

def _get_model_backbone(args):
    args.setdefault('reduction', 16)
    args.setdefault('attention_type', None)
    attention = args['attention_type']
    assert attention in ['softmax', 'se', 'cbam', 'self', None], \
        f"Attention: {attention} not in ['softmax', 'se', 'cbam', 'self', None]"

    if attention == 'softmax':
        return VGG19SoftmaxAttention()
    elif attention == 'se':
        return VGG19SEAttention(reduction=args['reduction'])
    elif attention == 'cbam':
        return VGG19CBAMAttention()
    elif attention == 'self':
        return VGG19SelfAttention()
    else:
        return VGG19Baseline()

def main():
    config = _parse_config()
    assert config['experiment'] == 'eval', \
        f"Experiment '{config['experiment']}' is not 'eval'"

    val_loader = _load_val_data(config)

    for model_name, args in config['models'].items():
        print(f"Model: {model_name}, args: {args}")
        ckpt_path = args['ckpt_path']

        backbone = _get_model_backbone(args)
        model = LightningWrapper.load_from_checkpoint(ckpt_path, model=backbone)

        acc = _evaluate(model, val_loader)
        print(f"Model Name: {model_name}, Checkpoint: {ckpt_path}, Val accuracy: {acc:.4f}")
        print("=" * 70)

if __name__ == '__main__':
    main()