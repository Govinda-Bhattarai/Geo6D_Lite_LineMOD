import os
import glob
import json
import yaml
import cv2
import torch

import numpy as np
from torch.utils.data import Dataset


def list_frames(rgb_dir):
    """List numeric PNG/JPG filenames in sorted order."""
    frames = []
    for ext in ["*.png", "*.jpg"]:
        frames.extend(
            [
                int(os.path.splitext(os.path.basename(p))[0])
                for p in sorted(glob.glob(os.path.join(rgb_dir, ext)))
                if os.path.splitext(os.path.basename(p))[0].isdigit()
            ]
        )
    return sorted(set(frames))


def load_split_file(split_file):
    """Load frame IDs from train.txt or test.txt."""
    if not os.path.isfile(split_file):
        return []
    try:
        with open(split_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Filter and convert: handle lines with leading zeros like "0009"
        frame_ids = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped.isdigit():
                frame_ids.append(int(stripped))
        return frame_ids
    except Exception as e:
        print(f"[WARN] Error loading split file {split_file}: {e}")
        return []


def bbox_from_mask(mask_np, min_size=8, pad=0.10):
    """Computes a padded bounding box around a binary mask."""
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if w < min_size or h < min_size:
        return None
    px, py = int(pad * w), int(pad * h)
    return max(0, x1 - px), max(0, y1 - py), x2 + px, y2 + py


def adjust_K_for_crop_and_resize(K, crop_box, out_size=256):
    """Adjusts the camera intrinsic matrix when cropped and resized."""
    (x1, y1, x2, y2) = crop_box
    crop_w, crop_h = (x2 - x1 + 1), (y2 - y1 + 1)
    sx = out_size / crop_w
    sy = out_size / crop_h
    K = K.copy()
    K[0, 2] = (K[0, 2] - x1) * sx
    K[1, 2] = (K[1, 2] - y1) * sy
    K[0, 0] *= sx
    K[1, 1] *= sy
    return K


class LineMODDriveMini(Dataset):
    """
    LineMOD dataset loader supporting RGB, Depth, Mask, and intrinsics.
    Works with the preprocessed LineMOD format using YAML files.
    Each sample contains:
        {
            'image': (3,H,W) tensor [0,1],
            'depth': (1,H,W) tensor,
            'mask': (1,H,W) tensor,
            'R': (3,3),
            't': (3,),
            'K': (3,3)
        }

    Args:
        data_root: Root directory containing object folders (e.g., "datasets/Linemod_preprocessed/data")
                  OR can be None to use config-based paths via object_ids
        objects: List of object names (e.g., ["ape", "can"]) OR object IDs (e.g., ["05", "09"])
        object_ids: List of object IDs (e.g., ["05", "09"]). If provided, uses cfg.get_linemod_paths()
        split: "train" or "test" to use train.txt or test.txt
        max_per_obj: Maximum number of samples per object
        img_size: Target image size
    """

    def __init__(
        self,
        data_root=None,
        objects=None,
        object_ids=None,
        split="train",
        max_per_obj=200,
        img_size=256,
        use_augmentation=False,
        validate_files=False,
    ):
        self.samples = []
        self.img_size = img_size
        self.use_augmentation = (
            use_augmentation and split == "train"
        )  # Only augment training data

        # Import config here to avoid circular imports
        from config import cfg

        # Determine which objects to process
        if object_ids is not None:
            # Use config-based paths
            obj_list = object_ids
            use_config_paths = True
        elif objects is not None:
            # Use provided objects (assume they are object IDs if data_root is None)
            obj_list = objects
            use_config_paths = data_root is None
        else:
            raise ValueError("Either 'objects' or 'object_ids' must be provided")

        for obj in obj_list:
            if use_config_paths:
                # Use config-based paths
                paths = cfg.get_linemod_paths(obj)
                obj_dir = paths["DATASET_ROOT"]
                rgb_dir = paths["RGB_DIR"]
                depth_dir = paths["DEPTH_DIR"]
                mask_dir = paths["MASK_DIR"]
                gt_path = paths["GT_FILE"]
                info_path = paths["INFO_FILE"]
                split_file = (
                    paths["TRAIN_SPLIT"] if split == "train" else paths["TEST_SPLIT"]
                )
            else:
                # Use provided data_root (legacy format)
                obj_dir = os.path.join(data_root, "train_pbr", obj)
                rgb_dir = os.path.join(obj_dir, "rgb")
                depth_dir = os.path.join(obj_dir, "depth")
                mask_dir = os.path.join(obj_dir, "mask_visib")
                gt_path = os.path.join(obj_dir, "scene_gt.json")
                info_path = os.path.join(obj_dir, "scene_camera.json")
                split_file = None

            if not os.path.isdir(rgb_dir):
                print(f"[WARN] Skipping {obj} (missing rgb directory: {rgb_dir})")
                continue

            # Load ground truth and camera info
            if use_config_paths:
                # Load YAML files
                if not os.path.isfile(gt_path) or not os.path.isfile(info_path):
                    print(f"[WARN] Skipping {obj} (missing YAML files)")
                    continue

                with open(gt_path, "r") as f:
                    gt = yaml.safe_load(f)
                with open(info_path, "r") as f:
                    cam = yaml.safe_load(f)

                # Get frame IDs from split file or list all frames
                # If max_per_obj is None, ignore split file and use all frames
                if (
                    split_file
                    and os.path.isfile(split_file)
                    and max_per_obj is not None
                ):
                    frame_ids = load_split_file(split_file)
                    # If split file is empty or couldn't be read, fall back to listing frames
                    if len(frame_ids) == 0:
                        print(
                            f"[INFO] Split file {split_file} is empty or unreadable, using all frames from RGB directory"
                        )
                        frame_ids = list_frames(rgb_dir)
                    else:
                        print(
                            f"[INFO] Using {len(frame_ids)} frames from split file {split_file}"
                        )
                else:
                    if max_per_obj is None:
                        print(
                            "[INFO] max_per_obj=None: Using ALL available frames (ignoring split file)"
                        )
                    frame_ids = list_frames(rgb_dir)
            else:
                # Load JSON files (legacy format)
                if not os.path.isfile(gt_path) or not os.path.isfile(info_path):
                    print(f"[WARN] Skipping {obj} (missing JSON files)")
                    continue

                gt = json.load(open(gt_path))
                cam = json.load(open(info_path))
                frame_ids = list_frames(rgb_dir)

            used = 0
            for fid in frame_ids:
                # YAML files use integer keys, so use fid directly as int
                key = fid if use_config_paths else str(fid)

                if use_config_paths:
                    # YAML format: gt[key] is a list, cam[key] is a dict (keys are integers)
                    if key not in gt or key not in cam:
                        continue
                    ann_list = gt[key]
                    if not ann_list or len(ann_list) == 0:
                        continue
                    ann = ann_list[0]  # Take first annotation
                    Rm = np.array(ann["cam_R_m2c"]).reshape(3, 3)
                    t = np.array(ann["cam_t_m2c"]).reshape(3) / 1000.0  # mm → m
                    cam_K = cam[key]["cam_K"]
                    K = np.array(cam_K).reshape(3, 3)

                    rgb = os.path.join(rgb_dir, f"{fid:04d}.png")
                    depth = os.path.join(depth_dir, f"{fid:04d}.png")
                    mask = os.path.join(mask_dir, f"{fid:04d}.png")
                else:
                    # JSON format (legacy)
                    if key not in gt or key not in cam:
                        continue
                    ann = gt[key][0]
                    Rm = np.array(ann["cam_R_m2c"]).reshape(3, 3)
                    t = np.array(ann["cam_t_m2c"]).reshape(3) / 1000.0  # mm → m
                    K = np.array(cam[key]["cam_K"]).reshape(3, 3)

                    rgb = os.path.join(rgb_dir, f"{fid:06d}.jpg")
                    depth = os.path.join(depth_dir, f"{fid:06d}.png")
                    mask = os.path.join(mask_dir, f"{fid:06d}_000000.png")

                # Verify RGB file exists
                if not os.path.isfile(rgb):
                    continue

                # Optional quick check: try to read the file to ensure it's valid.
                # This check can be expensive for large datasets; callers may set
                # `validate_files=True` during debugging but leave it False for
                # faster initialization (default).
                if validate_files:
                    test_img = cv2.imread(rgb)
                    if test_img is None:
                        # File exists but can't be read (might be corrupted)
                        continue

                self.samples.append(
                    {
                        "rgb": rgb,
                        "depth": depth if os.path.isfile(depth) else None,
                        "mask": mask if os.path.isfile(mask) else None,
                        "R": Rm,
                        "t": t,
                        "K": K,
                        "object_id": str(obj),
                    }
                )

                used += 1
                # If max_per_obj is None, do not limit samples (use all available frames).
                if (max_per_obj is not None) and (used >= max_per_obj):
                    break

        print(f"✅ Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Verify RGB file exists before trying to read
        if not os.path.isfile(s["rgb"]):
            # If file doesn't exist, try to find a valid sample
            # This can happen if files were deleted after dataset initialization
            for i in range(len(self.samples)):
                alt_idx = (idx + i) % len(self.samples)
                alt_sample = self.samples[alt_idx]
                if os.path.isfile(alt_sample["rgb"]):
                    s = alt_sample
                    break
            else:
                raise FileNotFoundError(
                    f"RGB not found: {s['rgb']} (and no alternative found)"
                )

        img_bgr = cv2.imread(s["rgb"])
        if img_bgr is None:
            raise FileNotFoundError(
                f"RGB file exists but could not be read: {s['rgb']}"
            )
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        depth_t = None
        mask_t = None
        mask_full = None

        if s.get("mask") and os.path.isfile(s["mask"]):
            m = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mask_full = (m > 128).astype(np.uint8)
            else:
                mask_full = None

        # Crop if mask exists
        if mask_full is not None:
            box = bbox_from_mask(mask_full)
            if box is not None:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W0 - 1, x2), min(H0 - 1, y2)
                img = img[y1 : y2 + 1, x1 : x2 + 1]
                if s.get("depth") and os.path.isfile(s["depth"]):
                    d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED)
                    if d is not None:
                        d = d.astype(np.float32) / 1000.0
                        d = d[y1 : y2 + 1, x1 : x2 + 1]
                    else:
                        d = None
                else:
                    d = None
                if mask_full is not None:
                    mask_crop = mask_full[y1 : y2 + 1, x1 : x2 + 1]
                else:
                    mask_crop = None
                K_adj = adjust_K_for_crop_and_resize(
                    s["K"].astype(np.float32), (x1, y1, x2, y2), out_size=self.img_size
                )
            else:
                K_adj = s["K"].astype(np.float32)
                if s.get("depth") and os.path.isfile(s["depth"]):
                    d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED)
                    d = d.astype(np.float32) / 1000.0 if d is not None else None
                else:
                    d = None
                mask_crop = mask_full
        else:
            K_adj = s["K"].astype(np.float32)
            if s.get("depth") and os.path.isfile(s["depth"]):
                d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED)
                d = d.astype(np.float32) / 1000.0 if d is not None else None
            else:
                d = None
            mask_crop = mask_full

        # Apply data augmentation (only for training)
        if self.use_augmentation:
            try:
                from utils.augmentation import augment_sample

                # Get current image dimensions (after crop)
                h_crop, w_crop = img.shape[:2]
                # Convert to [0, 1] range for augmentation
                img_norm = img.astype(np.float32) / 255.0
                # Prepare depth and mask (ensure correct shape)
                d_aug = (
                    d if d is not None else np.zeros((h_crop, w_crop), dtype=np.float32)
                )
                mask_aug = (
                    mask_crop
                    if mask_crop is not None
                    else np.zeros((h_crop, w_crop), dtype=np.uint8)
                )
                # Augment
                img_norm, d_aug, mask_aug, K_adj, s["R"], s["t"] = augment_sample(
                    img_norm,
                    d_aug,
                    mask_aug,
                    K_adj,
                    s["R"],
                    s["t"],
                    apply_rot=True,
                    apply_color=True,
                )
                # Update d and mask_crop
                d = d_aug if d is not None else None
                mask_crop = mask_aug if mask_crop is not None else None
                # Convert back to [0, 255] for resize
                img = (img_norm * 255.0).astype(np.uint8)
            except Exception as e:
                # If augmentation fails, continue without it
                import traceback

                print(
                    f"[WARN] Augmentation failed: {e}, continuing without augmentation"
                )
                traceback.print_exc()

        # Resize to model input
        img = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if d is not None:
            d = cv2.resize(
                d, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
            )
            depth_t = torch.from_numpy(d[None]).float()
        else:
            # Create empty depth tensor if depth is not available
            depth_t = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        if mask_crop is not None:
            mask_crop = cv2.resize(
                mask_crop,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_t = torch.from_numpy(mask_crop[None]).float()
        else:
            # Create empty mask tensor if mask is not available
            mask_t = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        return {
            "img": img_t,
            "depth": depth_t,
            "mask": mask_t,
            "R": torch.from_numpy(s["R"]).float(),
            "t": torch.from_numpy(s["t"]).float(),
            "K": torch.from_numpy(K_adj).float(),
            "object_id": s.get(
                "object_id", None
            ),  # Include object_id for model points lookup
        }
