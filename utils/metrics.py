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


def add_score(R_gt, t_gt, R_pred, t_pred, model_points):
    """
    Compute ADD (Average Distance of model points) score.
    
    Args:
        R_gt: Ground truth rotation matrix (3, 3)
        t_gt: Ground truth translation vector (3,)
        R_pred: Predicted rotation matrix (3, 3)
        t_pred: Predicted translation vector (3,)
        model_points: Model points in object coordinates (N, 3)
    
    Returns:
        ADD score in meters (average distance between transformed points)
    """
    # Transform model points using ground truth pose
    P_gt = (R_gt @ model_points.T).T + t_gt  # (N, 3)
    
    # Transform model points using predicted pose
    P_pred = (R_pred @ model_points.T).T + t_pred  # (N, 3)
    
    # Compute average distance
    distances = np.linalg.norm(P_gt - P_pred, axis=1)  # (N,)
    add = np.mean(distances)
    
    return add


def compute_add_accuracy_metrics(add_scores, add_threshold=0.10):
    """
    Compute accuracy metrics based on ADD threshold.
    
    Args:
        add_scores: List of ADD scores in meters
        add_threshold: ADD accuracy threshold in meters (default: 0.10 = 10cm)
    
    Returns:
        dict with accuracy percentages and counts
    """
    add_scores = np.array(add_scores)
    
    # Count samples within threshold
    correct = np.sum(add_scores <= add_threshold)
    total = len(add_scores)
    
    return {
        "add_accuracy": float(correct / total * 100),  # Percentage
        "add_correct": int(correct),
        "total_samples": int(total),
        "add_threshold": float(add_threshold),
        "mean_add": float(np.mean(add_scores)),
        "median_add": float(np.median(add_scores)),
        "std_add": float(np.std(add_scores)),
    }
