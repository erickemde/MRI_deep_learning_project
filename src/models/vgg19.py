import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn, VGG19_BN_Weights


class SEAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        
        return x * y


class VGG19Model(nn.Module):
    def __init__(
        self,
        attention_type=None,
        num_classes=4,
        pretrained=True,
        reduction_ratio=16
    ):
        super().__init__()
        
        self.vgg = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        frozen_params = sum(p.numel() for p in self.vgg.parameters())
        print(f"[INFO] VGG19 frozen: {frozen_params/1e6:.1f}M params")
        
        if attention_type == 'se':
            self.use_attention = True
            self.features = self.vgg.features
            self.attention = SEAttention(512, reduction_ratio=reduction_ratio)
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            print("[INFO] Using SE Attention")
            
        elif attention_type is None:
            self.use_attention = False
            final_in_features = self.vgg.classifier[-1].in_features
            self.vgg.classifier[-1] = nn.Linear(final_in_features, num_classes)
            print("[INFO] No attention (Baseline)")
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] Trainable params: {trainable_params/1e6:.3f}M")
    
    def forward(self, x):
        if self.use_attention:
            x = self.features(x)
            x = self.attention(x)
            x = self.classifier(x)
        else:
            x = self.vgg(x)
        
        return x