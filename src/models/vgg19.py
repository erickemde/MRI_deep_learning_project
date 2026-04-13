import torchvision
from torch import nn
import torch

BATCH_NORM = {
    True: (torchvision.models.vgg19_bn, torchvision.models.VGG19_BN_Weights.DEFAULT),
    False: (torchvision.models.vgg19, torchvision.models.VGG19_Weights.DEFAULT)
}

# VGG Model Class
class VGGModel(nn.Module):
    def __init__(self, pretrained=True, batch_norm=True, freeze=True, self_attention=False):
        super().__init__()

        # model selection
        model, weights = BATCH_NORM[batch_norm]

        # initialize vgg model
        self.vgg = model(weights=weights if pretrained else None)

        # freeze parameter updates
        if freeze: 
            for param in self.vgg.parameters():
                param.requires_grad = False

        # replace final layer with 4 classes
        final_in_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier[-1] = nn.Linear(final_in_features, 4)

        # replace last 2 conv blocks with self-attention blocks
        if self_attention:
            raise NotImplementedError("Self-attention blocks not yet implemented")
    
    def forward(self, x):
        return self.vgg(x)
    

class SABlock(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Self-attention blocks not yet implemented")

    def forward(self, x):
        raise NotImplementedError("Self-attention blocks not yet implemented")