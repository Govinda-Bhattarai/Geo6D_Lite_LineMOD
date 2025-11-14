import os, glob, json, cv2, torch
import numpy as np
from torch.utils.data import Dataset


def list_frames(rgb_dir):
    """List numeric JPG filenames in sorted order."""
    return [
        int(os.path.splitext(os.path.basename(p))[0])
        for p in sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
        if os.path.basename(p).split('.')[0].isdigit()
    ]


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
    Each sample contains:
        {
            'image': (3,H,W) tensor [0,1],
            'depth': (1,H,W) tensor,
            'mask': (1,H,W) tensor,
            'R': (3,3),
            't': (3,),
            'K': (3,3)
        }
    """
    def __init__(self, data_root, objects, split="train_pbr", max_per_obj=200, img_size=256):
        self.samples = []
        self.img_size = img_size

        for obj in objects:
            obj_dir = os.path.join(data_root, split, obj)  # <-- include split here
            rgb_dir = os.path.join(obj_dir, "rgb")
            depth_dir = os.path.join(obj_dir, "depth")
            mask_dir = os.path.join(obj_dir, "mask_visib")
       


            gt_path = os.path.join(obj_dir, "scene_gt.json")
            cam_path = os.path.join(obj_dir, "scene_camera.json")

            if not (os.path.isdir(rgb_dir) and os.path.isfile(gt_path) and os.path.isfile(cam_path)):
                print(f"[WARN] Skipping {obj} (missing data)")
                continue

            gt = json.load(open(gt_path))
            cam = json.load(open(cam_path))
            frames = list_frames(rgb_dir)
            used = 0

            for fid in frames:
                key = str(fid)
                if key not in gt or key not in cam:
                    continue

                ann = gt[key][0]
                Rm = np.array(ann["cam_R_m2c"]).reshape(3, 3)
                t = np.array(ann["cam_t_m2c"]).reshape(3) / 1000.0  # mm → m
                K = np.array(cam[key]["cam_K"]).reshape(3, 3)

                rgb = os.path.join(rgb_dir, f"{fid:06d}.jpg")
                depth = os.path.join(depth_dir, f"{fid:06d}.png")
                mask = os.path.join(mask_dir, f"{fid:06d}_000000.png")

                if not os.path.isfile(rgb):
                    continue

                self.samples.append({
                    "rgb": rgb,
                    "depth": depth if os.path.isfile(depth) else None,
                    "mask": mask if os.path.isfile(mask) else None,
                    "R": Rm,
                    "t": t,
                    "K": K
                })

                used += 1
                if used >= max_per_obj:
                    break

        print(f"✅ Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_bgr = cv2.imread(s["rgb"])
        if img_bgr is None:
            raise FileNotFoundError(f"RGB not found: {s['rgb']}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        depth_t = None
        mask_t = None
        mask_full = None

        if s.get("mask") and os.path.isfile(s["mask"]):
            m = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
            mask_full = (m > 128).astype(np.uint8)

        # Crop if mask exists
        if mask_full is not None:
            box = bbox_from_mask(mask_full)
            if box is not None:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W0 - 1, x2), min(H0 - 1, y2)
                img = img[y1:y2+1, x1:x2+1]
                if s.get("depth") and os.path.isfile(s["depth"]):
                    d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32)
                    d = d[y1:y2+1, x1:x2+1]
                else:
                    d = None
                if mask_full is not None:
                    mask_crop = mask_full[y1:y2+1, x1:x2+1]
                else:
                    mask_crop = None
                K_adj = adjust_K_for_crop_and_resize(s["K"].astype(np.float32), (x1, y1, x2, y2), out_size=self.img_size)
            else:
                K_adj = s["K"].astype(np.float32)
                d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32) if s.get("depth") else None
                mask_crop = mask_full
        else:
            K_adj = s["K"].astype(np.float32)
            d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32) if s.get("depth") else None
            mask_crop = mask_full

        # Resize to model input
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if d is not None:
            d = cv2.resize(d, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            depth_t = torch.from_numpy(d[None]).float()
        if mask_crop is not None:
            mask_crop = cv2.resize(mask_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask_t = torch.from_numpy(mask_crop[None]).float()

        return {
            "img": img_t,
            "depth": depth_t,
            "mask": mask_t,
            "R": torch.from_numpy(s["R"]).float(),
            "t": torch.from_numpy(s["t"]).float(),
            "K": torch.from_numpy(K_adj).float(),
        }
