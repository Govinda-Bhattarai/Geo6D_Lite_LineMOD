import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class ResNetBackbone(nn.Module):
    """
    Improved ResNet backbone (ResNet18/34) with multi-scale feature extraction.
    Extracts features from multiple layers for richer representation.
    """

    def __init__(self, pretrained=True, out_channels=256, backbone_type="resnet34"):
        super().__init__()
        # Use modern weights API instead of deprecated pretrained parameter
        if backbone_type == "resnet34":
            if pretrained:
                weights = ResNet34_Weights.IMAGENET1K_V1
            else:
                weights = None
            net = resnet34(weights=weights)
        else:  # resnet18 (default fallback)
            if pretrained:
                weights = ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            net = resnet18(weights=weights)

        # Extract layers
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # 64 channels
        self.layer2 = net.layer2  # 128 channels
        self.layer3 = net.layer3  # 256 channels
        self.layer4 = net.layer4  # 512 channels

        # Multi-scale feature fusion
        # Project all layers to same dimension and fuse
        self.proj1 = nn.Conv2d(64, out_channels // 4, 1)
        self.proj2 = nn.Conv2d(128, out_channels // 4, 1)
        self.proj3 = nn.Conv2d(256, out_channels // 4, 1)
        self.proj4 = nn.Conv2d(512, out_channels // 4, 1)

        # Fusion layer with BatchNorm and ReLU
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels

        # CRITICAL FIX: Proper initialization for fusion layers
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for fusion layers."""
        for m in self.fusion.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

        # Initialize projection layers
        for proj in [self.proj1, self.proj2, self.proj3, self.proj4]:
            nn.init.kaiming_normal_(proj.weight, mode="fan_out", nonlinearity="relu")
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0)

    def forward(self, x):
        """
        Extract multi-scale features and fuse them.

        Args:
            x: (B, 3, H, W) input image

        Returns:
            feat: (B, out_channels, H/32, W/32) fused features
        """
        x = self.stem(x)  # H/4, W/4
        x1 = self.layer1(x)  # H/4, W/4, 64
        x2 = self.layer2(x1)  # H/8, W/8, 128
        x3 = self.layer3(x2)  # H/16, W/16, 256
        x4 = self.layer4(x3)  # H/32, W/32, 512

        # Project all to same dimension
        p1 = self.proj1(x1)  # H/4, W/4, out_channels//4
        p2 = self.proj2(x2)  # H/8, W/8, out_channels//4
        p3 = self.proj3(x3)  # H/16, W/16, out_channels//4
        p4 = self.proj4(x4)  # H/32, W/32, out_channels//4

        # Upsample and concatenate
        h, w = x4.shape[-2:]
        p1 = F.interpolate(p1, size=(h, w), mode="bilinear", align_corners=False)
        p2 = F.interpolate(p2, size=(h, w), mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=(h, w), mode="bilinear", align_corners=False)

        # Concatenate all features
        fused = torch.cat([p1, p2, p3, p4], dim=1)  # (B, out_channels, H/32, W/32)

        # Final fusion
        feat = self.fusion(fused)

        return feat
