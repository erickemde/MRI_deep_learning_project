import torch
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


class CustomChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        self.channel_weights = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        B, C, H, W = x.size()
        avg_out = self.avg_pool(x).view(B, C)
        avg_out = self.fc(avg_out)
        max_out = self.max_pool(x).view(B, C)
        max_out = self.fc(max_out)
        attention_weights = torch.sigmoid(avg_out + max_out)
        attention_weights = attention_weights * self.channel_weights.unsqueeze(0)
        out = x * attention_weights.view(B, C, 1, 1)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = CustomChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.query = nn.Conv2d(num_channels, num_channels // 8, 1)
        self.key   = nn.Conv2d(num_channels, num_channels // 8, 1)
        self.value = nn.Conv2d(num_channels, num_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B x (H*W) x C'
        key   = self.key(x).view(B, -1, H * W)                      # B x C' x (H*W)
        value = self.value(x).view(B, -1, H * W)                    # B x C x (H*W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)         # B x (H*W) x (H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))           # B x C x (H*W)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


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

        classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        if attention_type == 'softmax':
            self.use_attention = True
            self.features = self.vgg.features
            self.attention = SoftmaxAttention(512)
            self.classifier = classifier_head
            print("[INFO] Using Softmax Attention")

        elif attention_type == 'se':
            self.use_attention = True
            self.features = self.vgg.features
            self.attention = SEAttention(512, reduction_ratio=reduction_ratio)
            self.classifier = classifier_head
            print("[INFO] Using SE Attention")

        elif attention_type == 'cbam':
            self.use_attention = True
            self.features = self.vgg.features
            self.attention = CBAM(512, reduction_ratio=reduction_ratio)
            self.classifier = classifier_head
            print("[INFO] Using CBAM Attention")

        elif attention_type == 'self':
            self.use_attention = True
            self.features = self.vgg.features
            self.attention = SelfAttention(512)
            self.classifier = classifier_head
            print("[INFO] Using Self-Attention")

        elif attention_type is None:
            self.use_attention = False
            final_in_features = self.vgg.classifier[-1].in_features
            self.vgg.classifier[-1] = nn.Linear(final_in_features, num_classes)
            print("[INFO] No attention (Baseline)")

        else:
            raise ValueError(f"Invalid attention_type: {attention_type}")

        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params    = total_params - trainable_params

        print(f"[INFO] VGG19 frozen: {frozen_params/1e6:.1f}M params")
        print(f"[INFO] Trainable params: {trainable_params/1e6:.3f}M")

    def forward(self, x):
        if self.use_attention:
            x = self.features(x)
            x = self.attention(x)
            x = self.classifier(x)
        else:
            x = self.vgg(x)
        return x