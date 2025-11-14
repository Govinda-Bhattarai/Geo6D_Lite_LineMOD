import numpy as np

def bbox_from_mask(mask_np, min_size=8, pad=0.10):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if w < min_size or h < min_size:
        return None
    px, py = int(pad * w), int(pad * h)
    return max(0, x1 - px), max(0, y1 - py), x2 + px, y2 + py

def adjust_K_for_crop_and_resize(K, crop_box, in_size=None, out_size=256):
    (x1, y1, x2, y2) = crop_box
    crop_w, crop_h = (x2 - x1 + 1), (y2 - y1 + 1)
    sx = out_size / crop_w
    sy = out_size / crop_h
    K = K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - x1) * sx
    K[1, 2] = (K[1, 2] - y1) * sy
    return K
