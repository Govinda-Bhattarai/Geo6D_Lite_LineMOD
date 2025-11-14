import torch
import torch.nn as nn
import torch.nn.functional as F
from .rot6d import rot6d_to_matrix

class PoseHead(nn.Module):
    def __init__(self, in_c, geo_c, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c + geo_c, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_rot = nn.Linear(hidden, 6)
        self.fc_trans = nn.Linear(hidden, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc_rot(x), self.fc_trans(x)


class Geo6DNet(nn.Module):
    def __init__(self, backbone, geo_channels=10):
        super().__init__()
        self.backbone = backbone
        self.head = PoseHead(backbone.out_channels, geo_channels)

    def _build_geo(self, depth, K, feat_hw, img_hw, device):
        B = K.shape[0]
        H, W = img_hw
        hF, wF = feat_hw
        outs = []
        for b in range(B):
            ys, xs = torch.meshgrid(
                torch.linspace(0, H - 1, H, device=device),
                torch.linspace(0, W - 1, W, device=device),
                indexing='ij'
            )
            un = (xs / (W - 1) - 0.5) * 2
            vn = (ys / (H - 1) - 0.5) * 2
            chans = [xs, ys, un, vn]
            if depth is not None:
                d = depth[b, 0]
                fx, fy, cx, cy = K[b, 0, 0], K[b, 1, 1], K[b, 0, 2], K[b, 1, 2]
                X = (xs - cx) / fx * d
                Y = (ys - cy) / fy * d
                Z = d
                Xm, Ym, Zm = X.mean(), Y.mean(), Z.mean()
                chans += [X, Y, Z, X - Xm, Y - Ym, Z - Zm]
            geo = torch.stack(chans, 0).unsqueeze(0)
            geo = F.interpolate(geo, size=(hF, wF), mode='bilinear', align_corners=False)
            outs.append(geo.squeeze(0))
        return torch.stack(outs, 0)

    def forward(self, image, depth, K):
        feat = self.backbone(image)
        hF, wF = feat.shape[-2:]
        H, W = image.shape[-2:]
        device = image.device
        geo = self._build_geo(depth, K, (hF, wF), (H, W), device)
        fused = torch.cat([feat, geo], 1)
        rot6d, t = self.head(fused)
        R = rot6d_to_matrix(rot6d)
        return {"R": R, "t_off": t, "rot6d": rot6d}
