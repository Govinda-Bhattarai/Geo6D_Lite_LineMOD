import numpy as np

def rotation_error(R_gt, R_pred):
    """Angular difference in degrees between rotation matrices."""
    cos = (np.trace(R_gt.T @ R_pred) - 1) / 2
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))

def translation_error(t_gt, t_pred):
    """Euclidean distance in cm."""
    return np.linalg.norm(t_gt - t_pred) * 100

def rotation_accuracy(R_gt, R_pred, threshold=10.0):
    """Check if rotation error is within threshold (default: 10 degrees)."""
    err = rotation_error(R_gt, R_pred)
    return err <= threshold

def translation_accuracy(t_gt, t_pred, threshold=5.0):
    """Check if translation error is within threshold (default: 5 cm)."""
    err = translation_error(t_gt, t_pred)
    return err <= threshold

def compute_accuracy_metrics(rot_errors, trans_errors, rot_threshold=10.0, trans_threshold=5.0):
    """
    Compute accuracy metrics based on thresholds.
    
    Args:
        rot_errors: List of rotation errors in degrees
        trans_errors: List of translation errors in cm
        rot_threshold: Rotation accuracy threshold in degrees (default: 10°)
        trans_threshold: Translation accuracy threshold in cm (default: 5 cm)
    
    Returns:
        dict with accuracy percentages and counts
    """
    rot_errors = np.array(rot_errors)
    trans_errors = np.array(trans_errors)
    
    # Count samples within threshold
    rot_correct = np.sum(rot_errors <= rot_threshold)
    trans_correct = np.sum(trans_errors <= trans_threshold)
    
    # Count samples within both thresholds (overall accuracy)
    both_correct = np.sum((rot_errors <= rot_threshold) & (trans_errors <= trans_threshold))
    
    total = len(rot_errors)
    
    return {
        "rotation_accuracy": float(rot_correct / total * 100),  # Percentage
        "translation_accuracy": float(trans_correct / total * 100),  # Percentage
        "overall_accuracy": float(both_correct / total * 100),  # Percentage
        "rotation_correct": int(rot_correct),
        "translation_correct": int(trans_correct),
        "overall_correct": int(both_correct),
        "total_samples": int(total),
        "rotation_threshold": float(rot_threshold),
        "translation_threshold": float(trans_threshold),
    }
