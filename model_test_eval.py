import argparse
import torch
from torch.utils.data import DataLoader
from src.data.dataset import BrainTumorDataset, load_dataset_from_directory
from src.data.augmentation import get_val_transforms
import yaml
from src.models.LightningWrapper import LightningWrapper
from src.models.vgg19 import VGG19Baseline, VGG19SEAttention, VGG19SoftmaxAttention, VGG19CBAMAttention, VGG19SelfAttention, VGG19LoRA

torch.set_float32_matmul_precision('medium')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault('models', {})
    return config

def _load_test_data(config):
    test_paths, test_labels = load_dataset_from_directory(
        data_dir=config['data_dir'], split='test')
    test_dataset = BrainTumorDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=get_val_transforms())
    print(f"  Test images: {len(test_dataset)}")
    return DataLoader(test_dataset, batch_size=config["batch_size"],
                     shuffle=False, num_workers=config["num_workers"],
                     pin_memory=True)

def _evaluate(model, loader):
    model.eval()
    model.to(DEVICE)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            pred = model(images).argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total

def _get_model_backbone(args):
    args.setdefault('reduction', 16)
    attention = args.get('attention_type', None)
    if attention == 'softmax': return VGG19SoftmaxAttention()
    elif attention == 'se': return VGG19SEAttention(reduction=args['reduction'])
    elif attention == 'cbam': return VGG19CBAMAttention()
    elif attention == 'self': return VGG19SelfAttention()
    elif attention == 'lora': return VGG19LoRA(rank=args.get('rank', 8))
    else: return VGG19Baseline()

def main():
    config = _parse_config()
    test_loader = _load_test_data(config)
    print(f"\n{'='*60}")
    print(f"Data dir: {config['data_dir']}")
    print(f"{'='*60}")
    for model_name, args in config['models'].items():
        model = LightningWrapper.load_from_checkpoint(
            args['ckpt_path'], model=_get_model_backbone(args))
        acc = _evaluate(model, test_loader)
        print(f"{model_name}: Test Acc = {acc:.4f} ({acc*100:.2f}%)")
        print("="*60)

if __name__ == '__main__':
    main()
