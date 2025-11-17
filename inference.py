"""
Production-ready inference wrapper for Geo6D-Lite pose estimation.
This module mirrors the training/evaluation pipeline so that deployment
code uses the exact same preprocessing (RGB + depth + intrinsics + mask).

Typical usage:

    from inference import PoseEstimator
    estimator = PoseEstimator("checkpoints/epoch_39.pth", object_id="05")
    rot6d, trans, R = estimator.predict(rgb_np, depth_np, intrinsics_np, mask_np)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from config import cfg
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from models.rot6d import rot6d_to_matrix


ArrayLike = Union[np.ndarray, torch.Tensor]


class PoseEstimator:
    """Deployment-friendly wrapper around Geo6DNet."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        object_id: str = "05",
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.object_id = str(object_id)
        self.input_res = getattr(cfg.model, "input_res", 256)
        self.requires_depth = getattr(cfg.model, "use_depth", True)

        backbone = ResNetBackbone(
            pretrained=cfg.model.pretrained,
            out_channels=cfg.model.feat_dim,
            backbone_type=cfg.model.backbone,
        )
        geo_channels = getattr(cfg.model, "geo_channels", 12)
        self.model = Geo6DNet(backbone, geo_channels=geo_channels).to(self.device)
        self.model.eval()

        state_dict = self._load_state_dict(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"✓ Model loaded from {checkpoint_path} onto {self.device}")

        self.default_intrinsics = self._load_default_intrinsics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        image: ArrayLike,
        depth: Optional[ArrayLike],
        intrinsics: Optional[ArrayLike] = None,
        mask: Optional[ArrayLike] = None,
        return_matrix: bool = True,
        return_dict: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], dict, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Run a forward pass of Geo6DNet.

        Args:
            image: RGB image in [0,1], shape (H,W,3) or tensor (3,H,W)
            depth: Depth map in meters (H,W) or (1,H,W)
            intrinsics: 3x3 camera matrix. Defaults to dataset intrinsics.
            mask: Foreground mask (H,W) or (1,H,W), optional but recommended.
            return_matrix: If True, include rotation matrix in the return tuple/dict.
            return_dict: If True, return a dict with all outputs.
        """

        rgb = self._prepare_rgb(image)
        spatial_size = rgb.shape[-2:]
        d_tensor = self._prepare_depth(depth, spatial_size)
        mask_tensor = self._prepare_mask(mask, spatial_size)
        K_tensor = self._prepare_intrinsics(intrinsics)

        with torch.no_grad():
            outputs = self.model(rgb, d_tensor, K_tensor, mask=mask_tensor)

        rot6d = outputs["rot6d"].squeeze(0).cpu().numpy()
        trans = outputs["t_off"].squeeze(0).cpu().numpy()
        rot_matrix = (
            rot6d_to_matrix(outputs["rot6d"]).squeeze(0).cpu().numpy() if return_matrix else None
        )

        if return_dict:
            result = {
                "rotation_6d": rot6d,
                "translation": trans,
                "rotation_matrix": rot_matrix,
                "raw": outputs,
            }
            return result

        if return_matrix:
            return rot6d, trans, rot_matrix
        return rot6d, trans

    def predict_from_paths(
        self,
        rgb_path: str,
        depth_path: str,
        intrinsics: Optional[ArrayLike] = None,
        mask_path: Optional[str] = None,
        return_matrix: bool = True,
    ):
        """Helper that loads RGB/depth/mask from disk before running predict()."""
        rgb = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
        depth_img = np.array(Image.open(depth_path)).astype(np.float32)
        if depth_img.ndim == 3:
            depth_img = depth_img[..., 0]
        if depth_img.max() > 10.0:
            depth_img /= 1000.0  # assume millimeters
        mask = None
        if mask_path:
            mask_np = np.array(Image.open(mask_path))
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            mask = (mask_np > 0).astype(np.float32)
        return self.predict(rgb, depth_img, intrinsics=intrinsics, mask=mask, return_matrix=return_matrix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_state_dict(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        # Allow loading raw state dicts
        if isinstance(ckpt, dict):
            return ckpt
        raise KeyError(f"Checkpoint at {checkpoint_path} does not contain a compatible state dict.")

    def _load_default_intrinsics(self) -> np.ndarray:
        try:
            paths = cfg.get_linemod_paths(self.object_id)
            info_path = paths.get("INFO_FILE")
            if info_path and os.path.isfile(info_path):
                with open(info_path, "r", encoding="utf-8") as f:
                    info = yaml.safe_load(f)
                if isinstance(info, dict) and len(info) > 0:
                    first_key = sorted(info.keys())[0]
                    cam_K = np.array(info[first_key]["cam_K"]).reshape(3, 3).astype(np.float32)
                    return cam_K
        except Exception as exc:
            print(f"⚠️  Could not load default intrinsics for object {self.object_id}: {exc}")
        return np.eye(3, dtype=np.float32)

    def _prepare_rgb(self, image: ArrayLike) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.clone().float()
        else:
            tensor = torch.from_numpy(np.asarray(image)).float()
        if tensor.ndim == 3:
            if tensor.shape[0] == 3:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError("RGB image must have 3 channels.")
        elif tensor.ndim == 4 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("RGB tensor must have shape (B,3,H,W) or (H,W,3).")

        if tensor.max() > 1.5:
            tensor = tensor / 255.0

        tensor = F.interpolate(
            tensor,
            size=(self.input_res, self.input_res),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.to(self.device)

    def _prepare_depth(self, depth: Optional[ArrayLike], spatial_size) -> Optional[torch.Tensor]:
        if depth is None:
            if self.requires_depth:
                raise ValueError("Depth input is required but was not provided.")
            return None

        if isinstance(depth, torch.Tensor):
            tensor = depth.clone().float()
        else:
            tensor = torch.from_numpy(np.asarray(depth)).float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[0] == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[-1] == 1:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError("Depth must have a single channel.")
        elif tensor.ndim == 4 and tensor.shape[1] == 1:
            pass
        elif tensor.ndim == 4 and tensor.shape[-1] == 1:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError("Depth tensor has unsupported shape.")

        if tensor.max() > 10.0:
            tensor = tensor / 1000.0  # assume millimeters

        tensor = F.interpolate(
            tensor,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        return tensor.to(self.device)

    def _prepare_mask(self, mask: Optional[ArrayLike], spatial_size) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        if isinstance(mask, torch.Tensor):
            tensor = mask.clone().float()
        else:
            tensor = torch.from_numpy(np.asarray(mask)).float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[0] == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[-1] == 1:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                tensor = (
                    tensor.mean(dim=-1, keepdim=True).permute(2, 0, 1).unsqueeze(0)
                )
        elif tensor.ndim == 4 and tensor.shape[1] == 1:
            pass
        elif tensor.ndim == 4 and tensor.shape[-1] == 1:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError("Mask must be single-channel.")

        tensor = (tensor > 0.5).float()
        tensor = F.interpolate(
            tensor,
            size=spatial_size,
            mode="nearest",
        )
        return tensor.to(self.device)

    def _prepare_intrinsics(self, intrinsics: Optional[ArrayLike]) -> torch.Tensor:
        if intrinsics is None:
            K = self.default_intrinsics
        else:
            K = np.asarray(intrinsics)
        if K.shape != (3, 3):
            raise ValueError("Camera intrinsics must be a 3x3 matrix.")
        tensor = torch.from_numpy(K).float().unsqueeze(0)
        return tensor.to(self.device)


if __name__ == "__main__":
    print("Geo6D-Lite Production Inference\n" + "=" * 50)
    estimator = PoseEstimator(
        "checkpoints/epoch_39.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        object_id="05",
    )
    rgb = np.random.rand(256, 256, 3).astype(np.float32)
    depth = np.ones((256, 256), dtype=np.float32)
    depth *= 1.0
    rot6d, trans, rot_mat = estimator.predict(rgb, depth, intrinsics=None, mask=None, return_matrix=True)
    print("Rotation (6D):", rot6d)
    print("Translation:", trans)
    print("Rotation matrix:\n", rot_mat)
