import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm
import argparse

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.device = next(self.model.parameters()).device
        self._register_hooks()

    def _register_hooks(self):
        """
        set up hooks to save activations and gradients at target layer
        """
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        """
        Generate gradcam heatmap
        """
        # set model to eval and place input to same device
        self.model.eval()
        x = x.to(self.device)

        # make sure parameters are unfrozen
        current_grad = [p.requires_grad for p in self.model.parameters()]
        for param in self.model.parameters():
            param.requires_grad = True

        try:
            logits = self.model(x)

            # if no class provided, use predicted class
            if class_idx is None:
                class_idx=logits.argmax(dim=1).item()

            # backpropagation from target class
            self.model.zero_grad()
            logits[0, class_idx].backward()

            # calculate gradcam
            weights = self.gradients.mean(dim=[2,3], keepdim=True)
            cam = (weights*self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
        finally:
            for param, grad_status in zip(self.model.parameters(), current_grad):
                param.requires_grad = grad_status

        return cam, class_idx
    
    def visualize(self, img_path, class_idx=None, alpha=0.4):
        """
        Visualize gradcam heatmap on original image
        """

        # read image path
        img = Image.open(img_path).convert("RGB")
        # convert to tensor and apply model transformations
        x = self.model.transforms(img).unsqueeze(0)
        img = np.array(img.resize((x.shape[3], x.shape[2])))/255.0

        # generate heatmap
        cam, class_idx = self.generate(x, class_idx)

        # resize and normalize heatmap
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8) # added 1e-8 for stability

        # combine original image and heatmap
        heatmap = plt.cm.jet(cam)[:,:,:3]
        overlay = alpha*heatmap + (1-alpha)*img

        # plot original, heatmap, and overlay
        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[1].imshow(heatmap)
        axes[1].set_title("Heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")

        parent = Path(img_path).parent.name
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        if parent in classes:
            fig.suptitle(f"True: {parent} | Pred: {classes[class_idx]}")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        return fig
    
    def examples(
            self, 
            dataloader, 
            save_dir="examples", 
            total_examples = 3, 
            seed=42
        ):
        """
        Summarize Key Grad Cam Results:
        1. Generate Grad Cam Examples for each Class
        2. Compute Confusion Matrix for Model
        3. Generate Grad Cam Examples for misclassified examples
        """

        # set random seed and create save_dir
        random.seed(seed)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

        # gradcam examples per class
        print("="*60)
        print("Generating GradCAM Examples by Class")
        print("="*60)
        for cls in tqdm(classes):
            cls_dir = Path("data/val")/cls
            cls_img_paths = random.sample(list(cls_dir.glob("*.jpg")), total_examples)
            save_path = Path(save_dir)/"gradcam"/cls
            save_path.mkdir(parents=True, exist_ok=True)
            for i, cls_img_path in enumerate(cls_img_paths):
                fig = self.visualize(cls_img_path)
                fig.savefig(save_path/f"{cls}_{i}.png")
                plt.close(fig)
        

        # confusion matrix
        print("="*60)
        print("Generating Confusion Matrix")
        print("="*60)
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(self.device)
                preds = self.model(x).argmax(dim=1).cpu()
                all_preds.extend(preds)
                all_labels.extend(y)

        
        conf_matrix = confusion_matrix(all_labels, all_preds)
        display = ConfusionMatrixDisplay(conf_matrix, display_labels=classes)
        fig, ax = plt.subplots(figsize=(8,8))
        display.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        plt.close()

        # misclassified examples
        print("="*60)
        print("Generating Misclassified Examples")
        print("="*60)

        misclassified_dir = Path(save_dir)/"misclassified"
        misclassified_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(Path("data/val").rglob("*.jpg"))
        random.shuffle(image_paths)
        total_wrong = 0
        for path in image_paths:
            if total_wrong>=total_examples:
                break
            true_label = path.parent.name
            fig = self.visualize(path)
            pred = fig.get_suptitle().split("Pred: ")[1]
            if pred!=true_label:
                fig.savefig(misclassified_dir/f"{true_label}_as_{pred}_{total_wrong}.png")
                total_wrong+=1
                print("Total Wrong:", total_wrong)
            plt.close(fig)

        print(f"Done! Results saved to {save_dir}")