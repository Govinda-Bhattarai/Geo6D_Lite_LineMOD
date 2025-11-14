import torch
import torch.nn as nn
from models.backbone import ResNetBackbone
from models.pose_head import PoseHead

class Geo6DModel(nn.Module):
    def __init__(self, feat_dim=256, pretrained=True):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained, out_channels=feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = PoseHead(feat_dim=feat_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        rot, trans = self.head(feat)
        return rot, trans
