import torch
import torch.nn as nn
import torch.nn.functional as F
from .rot6d import rot6d_to_matrix


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm for better training."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        # CRITICAL FIX: Proper initialization for residual blocks
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for better training stability."""
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(
                self.skip.weight, mode="fan_out", nonlinearity="relu"
            )
        # BatchNorm: weight=1, bias=0 (default is correct)
        if self.bn1.weight is not None:
            nn.init.constant_(self.bn1.weight, 1.0)
            nn.init.constant_(self.bn1.bias, 0.0)
        if self.bn2.weight is not None:
            nn.init.constant_(self.bn2.weight, 1.0)
            nn.init.constant_(self.bn2.bias, 0.0)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class PoseHead(nn.Module):
    """
    Improved Geo6D-style dense pose head with residual connections.
    Produces dense maps (rot6d_map, trans_map, conf_map) and computes
    global rot6d and translation by confidence-weighted pooling.
    """

    def __init__(self, in_c, geo_c, hidden=512):  # Increased from 256 to 512
        super().__init__()

        # Input fusion layer
        self.input_fusion = nn.Sequential(
            nn.Conv2d(in_c + geo_c, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # Deeper network with residual connections (increased from 4 to 6 blocks)
        self.conv_block = nn.Sequential(
            ResidualBlock(hidden, hidden),
            ResidualBlock(hidden, hidden),
            ResidualBlock(hidden, hidden),
            ResidualBlock(hidden, hidden),
            ResidualBlock(hidden, hidden),  # NEW: Added 5th block
            ResidualBlock(hidden, hidden),  # NEW: Added 6th block
        )

        # Output heads with better initialization
        self.rot_conv = nn.Conv2d(hidden, 6, 1)
        self.trans_conv = nn.Conv2d(hidden, 3, 1)
        self.conf_conv = nn.Conv2d(hidden, 1, 1)

        # Better initialization for pose estimation
        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization for pose estimation."""
        # Rotation head - small weights for stability
        nn.init.normal_(self.rot_conv.weight, std=0.001)
        if self.rot_conv.bias is not None:
            nn.init.constant_(self.rot_conv.bias, 0.0)

        # CRITICAL FIX: Translation head needs larger initialization
        # Translation values are typically in range [-0.2, 0.2] for X/Y, [0.5, 1.5] for Z
        # Small std=0.001 makes it hard to learn - increase to 0.01
        nn.init.normal_(self.trans_conv.weight, std=0.01)
        if self.trans_conv.bias is not None:
            # Initialize bias to small positive Z (typical depth ~1.0m)
            # This helps model start closer to correct depth
            nn.init.constant_(self.trans_conv.bias[0], 0.0)  # X
            nn.init.constant_(self.trans_conv.bias[1], 0.0)  # Y
            nn.init.constant_(
                self.trans_conv.bias[2], 0.0
            )  # Z (or could be 0.5 for depth hint)

        # Confidence head - start with low confidence
        nn.init.normal_(self.conf_conv.weight, std=0.01)
        if self.conf_conv.bias is not None:
            nn.init.constant_(self.conf_conv.bias, -2.0)

    def forward(self, x, mask=None):
        """
        x: fused features (B, C, H, W)
        mask: optional binary mask (B,1,H,W) to zero background before pooling

        Returns:
            rot6d_global (B,6), trans_global (B,3), R (B,3,3), aux dict with dense maps
        """
        B = x.shape[0]

        # Input fusion
        feat = self.input_fusion(x)

        # Deep feature extraction with residuals
        feat = self.conv_block(feat)  # (B, hidden, H, W)

        # Generate dense predictions
        rot6d_map = self.rot_conv(feat)  # (B,6,H,W)
        trans_map = self.trans_conv(feat)  # (B,3,H,W)
        conf_map = torch.sigmoid(self.conf_conv(feat))  # (B,1,H,W)

        # Apply mask to confidence map (if provided)
        if mask is not None:
            # Resize mask to match feature/conf_map spatial resolution
            try:
                mask_resized = F.interpolate(
                    mask.float(), size=conf_map.shape[-2:], mode="nearest"
                )
            except Exception:
                mask_resized = mask.float()
            # Zero out background confidence
            conf_map = conf_map * (mask_resized > 0).float()

        # Flatten spatial dims
        rot_flat = rot6d_map.view(B, 6, -1)  # (B,6,N)
        trans_flat = trans_map.view(B, 3, -1)  # (B,3,N)
        conf_flat = conf_map.view(B, 1, -1)  # (B,1,N)

        # Confidence-weighted pooling (Geo6D style)
        w = conf_flat + 1e-6  # Avoid divide-by-zero
        w_sum = w.sum(-1, keepdim=True)  # (B,1,1)
        rot6d_global = (rot_flat * w).sum(-1) / w_sum.squeeze(-1)  # (B,6)
        trans_global = (trans_flat * w).sum(-1) / w_sum.squeeze(-1)  # (B,3)

        # Convert rot6d to rotation matrix
        R = rot6d_to_matrix(rot6d_global)

        aux = {"rot6d_map": rot6d_map, "trans_map": trans_map, "conf_map": conf_map}
        return rot6d_global, trans_global, R, aux


class Geo6DNet(nn.Module):
    """
    Improved Geo6D network with better feature fusion and geometric features.
    """

    def __init__(self, backbone, geo_channels=12):
        super().__init__()
        self.backbone = backbone
        # geo_channels should match what _build_geo produces:
        # With depth: 4 (coords) + 8 (3D features) = 12 channels
        self.head = PoseHead(backbone.out_channels, geo_channels)
        self.geo_channels = geo_channels

    def _build_geo(self, depth, K, feat_hw, img_hw, device):
        """
        Build geometric features from depth and camera intrinsics.
        Improved version with better normalization and more features.
        """
        B = K.shape[0]
        H, W = img_hw
        hF, wF = feat_hw
        outs = []

        for b in range(B):
            # Create pixel coordinate grids
            ys, xs = torch.meshgrid(
                torch.linspace(0, H - 1, H, device=device),
                torch.linspace(0, W - 1, W, device=device),
                indexing="ij",
            )

            # Normalized coordinates [-1, 1]
            un = (xs / (W - 1) - 0.5) * 2
            vn = (ys / (H - 1) - 0.5) * 2

            chans = [xs, ys, un, vn]  # 4 channels: pixel coords + normalized coords

            if depth is not None:
                d = depth[b, 0]  # (H, W)

                # Get camera intrinsics
                fx, fy, cx, cy = K[b, 0, 0], K[b, 1, 1], K[b, 0, 2], K[b, 1, 2]

                # Convert to 3D camera coordinates
                X = (xs - cx) / fx * d  # (H, W)
                Y = (ys - cy) / fy * d  # (H, W)
                Z = d  # (H, W)

                # Normalize 3D coordinates (subtract mean)
                Xm, Ym, Zm = X.mean(), Y.mean(), Z.mean()
                X_norm = X - Xm
                Y_norm = Y - Ym
                Z_norm = Z - Zm

                # Additional geometric features
                # Distance from camera center
                dist = torch.sqrt(X**2 + Y**2 + Z**2)
                dist_mean = dist.mean()
                dist_std = dist.std() + 1e-6  # Avoid division by zero
                dist_norm = (
                    dist - dist_mean
                ) / dist_std  # CRITICAL FIX: Proper normalization

                # CRITICAL FIX: Normalize all 3D coordinates properly
                # Normalize X, Y, Z by their standard deviations
                X_std = X.std() + 1e-6
                Y_std = Y.std() + 1e-6
                Z_std = Z.std() + 1e-6
                X_scaled = X / X_std
                Y_scaled = Y / Y_std
                Z_scaled = Z / Z_std

                chans += [
                    X_scaled,
                    Y_scaled,
                    Z_scaled,  # Normalized 3D coordinates (3 channels)
                    X_norm,
                    Y_norm,
                    Z_norm,  # Mean-centered 3D coordinates (3 channels)
                    dist / (dist_mean + 1e-6),  # Normalized distance (1 channel)
                    dist_norm,  # Standardized distance (1 channel)
                ]  # Total: 4 + 3 + 3 + 2 = 12 channels

            # Stack channels and interpolate to feature resolution
            geo = torch.stack(chans, 0).unsqueeze(0)  # (1, C, H, W)
            geo = F.interpolate(
                geo, size=(hF, wF), mode="bilinear", align_corners=False
            )
            outs.append(geo.squeeze(0))

        return torch.stack(outs, 0)  # (B, C, hF, wF)

    def forward(self, image, depth, K, mask=None):
        """
        Forward pass.

        Args:
            image: (B,3,H,W) RGB image [0,1]
            depth: (B,1,H,W) depth map in meters
            K: (B,3,3) camera intrinsics
            mask: (B,1,H,W) optional foreground mask [0,1]. If None, uses RGB-based mask.

        Returns:
            dict with R, t_off, rot6d, and dense maps
        """
        # Extract RGB features
        feat = self.backbone(image)  # (B, C, hF, wF)
        hF, wF = feat.shape[-2:]
        H, W = image.shape[-2:]
        device = image.device

        # Build geometric features
        geo = self._build_geo(depth, K, (hF, wF), (H, W), device)

        # Fuse RGB and geometric features
        fused = torch.cat([feat, geo], dim=1)  # (B, C+geo_channels, hF, wF)

        # Use provided mask if available, otherwise fall back to RGB-based mask
        if mask is None:
            # Fallback: Build a loose foreground mask from the RGB (non-empty pixels)
            mask = (image.mean(1, keepdim=True) > 0.01).float()
        else:
            # Use the provided mask (should be GT mask from dataset)
            mask = mask.float()

        # Forward through pose head
        rot6d, t, R, aux = self.head(fused, mask=mask)

        out = {"R": R, "t_off": t, "rot6d": rot6d}
        out.update(aux)
        return out
