import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn, VGG19_BN_Weights


class CustomChannelAttention(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        # Learnable weights w in R^C
        self.weights = nn.Parameter(torch.randn(num_channels))
        
    def forward(self, x):

        B, C, H, W = x.size()
        
        # Step 1: Global Average Pooling (Equation 1)
        v = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)
        
        # Step 2: SoftMax-weighted attention (Equation 2)
        alpha = F.softmax(self.weights, dim=0)  # (C,)
        
        # Step 3: Element-wise multiplication
        v_att = v * alpha  # Broadcasting: (B, C) * (C,)
        
        return v_att


class VGG19Model(nn.Module):

    def __init__(
        self, 
        pretrained=True, 
        freeze=True, 
        use_attention=True,
        num_classes=4
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Load VGG19 with Batch Normalization
        weights = VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg19_bn(weights=weights)
        
        # Extract feature layers (conv + bn + relu layers)
        self.features = vgg.features  # Output: (B, 512, 7, 7) for 224x224 input
        
        # Freeze backbone if requested
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Add attention mechanism
        if use_attention:
            # Custom Channel Attention
            self.attention = CustomChannelAttention(num_channels=512)
            # Compact classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            # Baseline: No attention
            self.attention = None
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 1024),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )
    
    def forward(self, x):

        # Feature extraction
        features = self.features(x)  # (B, 512, 7, 7)
        
        # Apply attention if enabled
        if self.use_attention:
            # Custom Attention: (B, 512, 7, 7) -> (B, 512)
            v_att = self.attention(features)
            output = self.classifier(v_att)
        else:
            # Baseline: Direct classification
            output = self.classifier(features)
        
        return output
    
    def get_attention_weights(self):

        if self.use_attention:
            # Return SoftMax-normalized weights
            return F.softmax(self.attention.weights, dim=0).detach()
        else:
            return None


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Test with attention
    model_with_attn = VGG19Model(use_attention=True)
    output = model_with_attn(x)
    print(f"VGG19 + Custom Attention - Output: {output.shape}")
    
    weights = model_with_attn.get_attention_weights()
    if weights is not None:
        print(f"Attention weights: {weights.shape}")
    
    # Test baseline (no attention)
    model_baseline = VGG19Model(use_attention=False)
    output = model_baseline(x)
    print(f"VGG19 Baseline - Output: {output.shape}")