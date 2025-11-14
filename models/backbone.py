import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        net = resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer2, self.layer3, self.layer4 = net.layer2, net.layer3, net.layer4
        self.proj = nn.Conv2d(512, out_channels, 1)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.proj(x)
