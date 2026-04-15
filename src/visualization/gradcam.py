import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
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
        device = next(self.model.parameters()).device
        x = x.to(device)

        # make sure parameters are unfrozen
        for param in self.model.parameters():
            param.requires_grad = True
        logits = self.model(x)

        # if no class provided, use predicted class
        if class_idx is None:
            class_idx=logits.argmax(dim=1)

        # backpropagation from target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # calculate gradcam
        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (weights*self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

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