"""
Data augmentation utilities for pose estimation.
Applies augmentation to RGB, depth, mask, and adjusts poses accordingly.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random


def apply_rotation_augmentation(img, depth, mask, K, R_gt, t_gt, angle_range=20.0):
    """
    Apply rotation augmentation around Z-axis (in-plane rotation).
    
    Args:
        img: (H, W, 3) RGB image
        depth: (H, W) depth map
        mask: (H, W) mask
        K: (3, 3) camera intrinsics
        R_gt: (3, 3) rotation matrix
        t_gt: (3,) translation vector
        angle_range: Maximum rotation angle in degrees
    
    Returns:
        Augmented img, depth, mask, K, R_gt, t_gt
    """
    if random.random() > 0.5:  # 50% chance
        return img, depth, mask, K, R_gt, t_gt
    
    angle = random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    
    # Rotation matrix for image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image, depth, mask
    img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    depth_rot = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Adjust K matrix for rotation
    # Rotation around image center doesn't change K, but we need to account for it
    # Actually, in-plane rotation doesn't affect K matrix
    K_adj = K.copy()
    
    # Adjust rotation matrix (rotate around camera Z-axis)
    angle_rad = np.radians(angle)
    R_z = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    R_gt_adj = R_gt @ R_z.T  # Rotate the object in the opposite direction
    
    return img_rot, depth_rot, mask_rot, K_adj, R_gt_adj, t_gt


def apply_brightness_contrast_augmentation(img, brightness_range=0.2, contrast_range=0.2):
    """
    Apply brightness and contrast augmentation to RGB image.
    
    Args:
        img: (H, W, 3) RGB image [0, 1]
        brightness_range: Maximum brightness change
        contrast_range: Maximum contrast change
    
    Returns:
        Augmented image
    """
    if random.random() > 0.5:  # 50% chance
        return img
    
    # Brightness
    brightness = random.uniform(-brightness_range, brightness_range)
    img = np.clip(img + brightness, 0, 1)
    
    # Contrast
    contrast = random.uniform(1 - contrast_range, 1 + contrast_range)
    img = np.clip(img * contrast, 0, 1)
    
    return img


def apply_color_jitter(img, jitter_range=0.1):
    """
    Apply color jitter (slight RGB channel shifts).
    
    Args:
        img: (H, W, 3) RGB image [0, 1]
        jitter_range: Maximum channel shift
    
    Returns:
        Augmented image
    """
    if random.random() > 0.5:  # 50% chance
        return img
    
    jitter = np.random.uniform(-jitter_range, jitter_range, 3)
    img = np.clip(img + jitter[None, None, :], 0, 1)
    
    return img


def augment_sample(img, depth, mask, K, R_gt, t_gt, apply_rot=True, apply_color=True):
    """
    Apply all augmentations to a sample.
    
    Args:
        img: (H, W, 3) RGB image [0, 1]
        depth: (H, W) depth map
        mask: (H, W) mask
        K: (3, 3) camera intrinsics
        R_gt: (3, 3) rotation matrix
        t_gt: (3,) translation vector
        apply_rot: Whether to apply rotation augmentation
        apply_color: Whether to apply color augmentation
    
    Returns:
        Augmented img, depth, mask, K, R_gt, t_gt
    """
    if apply_rot:
        angle_range = 20.0  # Can be overridden by config
        img, depth, mask, K, R_gt, t_gt = apply_rotation_augmentation(
            img, depth, mask, K, R_gt, t_gt, angle_range=angle_range
        )
    
    if apply_color:
        brightness_range = 0.2  # Can be overridden by config
        contrast_range = 0.2
        img = apply_brightness_contrast_augmentation(img, brightness_range=brightness_range, contrast_range=contrast_range)
        img = apply_color_jitter(img, jitter_range=0.05)
    
    return img, depth, mask, K, R_gt, t_gt

