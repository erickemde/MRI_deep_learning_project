import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import BrainTumorDataset, load_dataset_from_directory
from src.data.augmentation import get_val_transforms
import yaml
from src.models.LightningWrapper import LightningWrapper
from src.models.vgg19 import VGG19Baseline, VGG19SEAttention, VGG19SoftmaxAttention, VGG19CBAMAttention, VGG19SelfAttention
from src.visualization.gradcam import GradCAM


from huggingface_hub import hf_hub_download

torch.set_float32_matmul_precision('medium')


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _parse_config():
    '''
    Parse config and return the configuration of an experiment or ablation study
    '''
    parser = argparse.ArgumentParser(description='Brain Tumor Classification')    
    
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault('experiment', 'eval')
    config.setdefault('repo_id', 'gatech-deep-learning-project')
    config.setdefault('models', {})
    config.setdefault('cache_dir', '.cache/huggingface/')

    print("=" * 70)
    print(f"Config experiment: {config['experiment']}")
    print("=" * 70)

    return config

def _load_test_data(config):
    print("Loading test data...")

    test_paths, test_labels = load_dataset_from_directory(
        data_dir=config['data_dir'],
        split='test'
    )

    val_transform = get_val_transforms()

    test_dataset = BrainTumorDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    print(f"  Test batches: {len(test_loader)}")

    return test_loader

def _evaluate(model, test_loader):
    '''
    Evaluate final model on test set
    '''
    model.eval()
    model.to(DEVICE)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc

def _plot_confusion(all_labels, all_preds, model_name, save_dir="test_eval"):
    '''
    Generate a confusion matrix
    '''
    Path("test_eval").mkdir(exist_ok=True)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    conf_matrix = confusion_matrix(all_labels, all_preds)
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8,8))
    display.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.savefig(f"{save_dir}/confusion_matrix_{model_name}.png")
    plt.close()    
    print(f"Created confusion matrix for {model_name}")

def _evaluate_all(model, test_loader):
    '''
    Evaluate final model on test set
    '''
    model.eval()
    model.to(DEVICE)
    correct = 0
    total = 0
    all_pred = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            pred = model(images).argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            all_pred.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    return acc, all_pred, all_labels

def _download_hf_model(repo_id, model_name, ckpt_path, cache_dir):
    print(f"Downloading HuggingFace model: {repo_id}/{model_name}/{ckpt_path}...")
    return hf_hub_download(repo_id=f"{repo_id}/{model_name}", filename = ckpt_path, cache_dir = cache_dir)

def _get_model_backbone(args):
    args.setdefault('reduction', 16)
    args.setdefault('attention_type', None)
    attention = args['attention_type']
    assert attention in ['softmax', 'se', 'cbam', 'self', None], f"Attention: {attention} not in ['softmax', 'se', 'cbam', 'self', None]"

    if attention=='softmax':
        return VGG19SoftmaxAttention()
    elif attention=='se':
        return VGG19SEAttention(reduction=args['reduction'])
    elif attention=='cbam':
        return VGG19CBAMAttention()
    elif attention=='self':
        return VGG19SelfAttention()
    else:
        return VGG19Baseline()

def main():
    # Parse config
    config = _parse_config()
    assert config['experiment']=='eval', f"Experiment '{config['experiment']}' is not 'eval'"

    # Load data
    test_loader = _load_test_data(config)

    # Load and eval models
    for model_name, args in config['models'].items():
        print(f"Model: {model_name}, args: {args}")
        
        # Download checkpoint from huggingface
        ckpt_path = args['ckpt_path']
        ckpt = _download_hf_model(config['repo_id'], model_name, ckpt_path, config['cache_dir'])

        # Load model from checkpoint
        backbone = _get_model_backbone(args)
        model = LightningWrapper.load_from_checkpoint(ckpt, model=backbone)

        # Evaluate
        acc, all_preds, all_labels = _evaluate_all(model, test_loader)
        print(f"Model Name: {model_name}, Checkpoint: {ckpt_path}, Eval accuracy: {acc:.4f}")
        _plot_confusion(all_labels, all_preds, model_name)
        print("=" * 70)
        
        print("\n" + "=" * 70)
        print("GENERATING GRADCAM VISUALIZATIONS")
        print("=" * 70)

        try:
            gradcam_save_dir = os.path.join("gradcam_examples", "test", model_name, ckpt_path)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            gradcam = GradCAM(model, model.gradcam_target_layer)
            gradcam.examples(
                data_subset="test",
                save_dir=gradcam_save_dir,
                total_examples=config["total_examples"],
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                seed=config["seed"]
            )
        except Exception as e:
            print(f"[WARNING] GradCAM visualization failed: {e}")

if __name__ == '__main__':
    main()