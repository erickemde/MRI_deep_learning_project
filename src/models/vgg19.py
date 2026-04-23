import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn, VGG19_BN_Weights


class SoftmaxAttention(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_weights = nn.Parameter(torch.randn(num_channels))
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        v = self.avg_pool(x).view(B, C)
        alpha = F.softmax(self.channel_weights, dim=0)
        alpha = alpha.unsqueeze(0).expand(B, -1)
        
        out = x * alpha.view(B, C, 1, 1)
        return out


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
        reduction_ratio=16,
        unfreeze_from_layer=None
    ):
        super().__init__()
        
        self.vgg = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        if unfreeze_from_layer is not None:
            for i, layer in enumerate(self.vgg.features):
                if i >= unfreeze_from_layer:
                    for param in layer.parameters():
                        param.requires_grad = True
            print(f"[INFO] Unfroze layers from index {unfreeze_from_layer}")
        
        if attention_type == 'softmax':
            self.use_attention = True
            self.features = self.vgg.features
            self.attention = SoftmaxAttention(512)
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            print("[INFO] Using Softmax Attention")
            
        elif attention_type == 'se':
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
        else:
            raise ValueError(f"Invalid attention_type: {attention_type}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"[INFO] Total params: {total_params/1e6:.1f}M")
        print(f"[INFO] Frozen params: {frozen_params/1e6:.1f}M")
        print(f"[INFO] Trainable params: {trainable_params/1e6:.3f}M")
    
    def forward(self, x):
        if self.use_attention:
            x = self.features(x)
            x = self.attention(x)
            x = self.classifier(x)
        else:
            x = self.vgg(x)
        return x


class VGG19Baseline(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        self.model = VGG19Model(
            attention_type=None,
            num_classes=num_classes,
            pretrained=pretrained
        )

    def forward(self, x):
        return self.model(x)

    @property
    def gradcam_target_layer(self):
        """Returns the target layer for GradCAM visualization"""
        return self.model.vgg.features[-4]


class VGG19SEAttention(nn.Module):
    def __init__(self, num_classes=4, reduction=16, pretrained=True):
        super().__init__()
        self.num_classes = num_classes


        self.model = VGG19Model(
            attention_type='se',
            num_classes=num_classes,
            pretrained=pretrained,
            reduction_ratio=reduction
        )

    def forward(self, x):
        return self.model(x)

    @property
    def gradcam_target_layer(self):
        """Returns the target layer for GradCAM visualization"""
        return self.model.features[-4]


class VGG19SoftmaxAttention(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, unfreeze_from_layer=None):
        super().__init__()
        self.num_classes = num_classes

        self.model = VGG19Model(
            attention_type='softmax',
            num_classes=num_classes,
            pretrained=pretrained,
            unfreeze_from_layer=unfreeze_from_layer
        )

    def forward(self, x):
        return self.model(x)

    @property
    def gradcam_target_layer(self):
        """Returns the target layer for GradCAM visualization"""
        return self.model.features[-4]