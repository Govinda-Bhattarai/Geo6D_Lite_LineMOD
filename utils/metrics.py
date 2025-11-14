import numpy as np

def rotation_error(R_gt, R_pred):
    """Angular difference in degrees between rotation matrices."""
    cos = (np.trace(R_gt.T @ R_pred) - 1) / 2
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))

def translation_error(t_gt, t_pred):
    """Euclidean distance in cm."""
    return np.linalg.norm(t_gt - t_pred) * 100
