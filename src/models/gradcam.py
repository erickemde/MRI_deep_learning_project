import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        set up hooks to get activations and gradients for target layer
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
        self.model.eval()
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

        return cam
    
    # def visualize(self, x, img_path, class_idx=None, alpha=0.4):
    #     cam = self.generate(x, class_idx)

    #     cam = 