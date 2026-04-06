import torchvision
from torch import nn
import torch

# VGG Model Class
class VGGModel(nn.Module):
    def __init__(self, pretrained=True, batch_norm = True, freeze = True, self_attention=False):
        super().__init__()

        # download vgg model
        if batch_norm:
            model = torchvision.models.vgg19_bn
            weights = torchvision.models.VGG19_BN_Weights.DEFAULT
        else:
            model = torchvision.models.vgg19
            weights = torchvision.models.VGG19_Weights.DEFAULT

        # get weights for vgg model
        weights = weights if pretrained else None

        # initialize vgg model
        self.vgg = model(weights=weights)

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