from types import SimpleNamespace as NS
import os

cfg = NS(
    model=NS(
        backbone="resnet34",
        pretrained=False,
        use_depth=True,
        use_mask=True,
        feat_dim=512,  # Increased from 256 to 512 for ResNet34
        rot_repr="ortho6d",
        input_res=256,
        geo_channels=12,  # CRITICAL FIX: 4 (pixel coords) + 8 (3D features) = 12 channels
    ),
    loss=NS(
        # PHASE 3B: TUNED AUXILIARY ROTATION + REDUCED LR DECAY
        # Reduced aux rotation (0.04) to avoid overfitting; slower LR decay for smoother convergence
        w_rot=2.0,  # Stronger rotation weight (2x baseline)
        w_trans=2.0,  # Stronger translation weight (2x baseline)
        w_reproj=0.0,  # DISABLED: reprojection loss was distraction
        use_reproj=False,
        use_dense=False,
        # Dense supervision DISABLED - was competing with primary losses
        w_dense_rot=0.04,  # REDUCED: auxiliary rotation loss (was 0.08; overfitting likely)
        w_dense_trans=0.0,  # DISABLED: translation is already excellent
        w_conf=0.0,  # DISABLED: confidence learning off
    ),
    train=NS(
        lr=8e-5,  # Slightly higher for faster learning with stronger primary losses (ResNet34 + bigger head)
        lr_scheduler="step",  # Step scheduler for predictable decay
        lr_step_size=20,  # Step down every 20 epochs (was 12; slower decay to avoid overfitting)
        lr_gamma=0.8,  # Gentler decay (80% vs 70% to prevent aggressive LR drops)
        lr_warmup_epochs=2,  # 2 epoch warmup (was 3, reducing for faster convergence)
        lr_warmup_factor=0.3,  # Start at 30% of max LR
        lr_min=1e-6,  # Minimum LR floor
        wd=1e-4,  # Weight decay
        batch_size=8,  # Stable batch size for BatchNorm
        num_model_points=750,  # Balanced model points
        num_epochs=60,  # Target: 60 epochs (increased from 40 for ResNet34 convergence)
        use_augmentation=True,
        gradient_clip=1.0,  # Prevent gradient explosion
        # Moderate augmentation (helps generalization)
        aug_rotation_range=10.0,  # Moderate rotation augmentation
        aug_brightness_range=0.1,  # Moderate brightness augmentation
        aug_contrast_range=0.1,  # Moderate contrast augmentation
    ),
)

# attach local base paths to existing cfg object
cfg.BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _get_linemod_paths(object_id):
    obj = str(object_id)
    DATASET_ROOT = os.path.join(
        cfg.BASE_DIR, "datasets", "Linemod_preprocessed", "data", obj
    )
    return {
        "DATASET_ROOT": DATASET_ROOT,
        "RGB_DIR": os.path.join(DATASET_ROOT, "rgb"),
        "DEPTH_DIR": os.path.join(DATASET_ROOT, "depth"),
        "MASK_DIR": os.path.join(DATASET_ROOT, "mask"),
        "TRAIN_SPLIT": os.path.join(DATASET_ROOT, "train.txt"),
        "TEST_SPLIT": os.path.join(DATASET_ROOT, "test.txt"),
        "GT_FILE": os.path.join(DATASET_ROOT, "gt.yml"),
        "INFO_FILE": os.path.join(DATASET_ROOT, "info.yml"),
    }


cfg.get_linemod_paths = _get_linemod_paths

# sensible default local checkpoint location (can be overridden)
cfg.DEFAULT_CHECKPOINT = os.path.join(
    cfg.BASE_DIR, "checkpoints", "geo6d_lite_latest.pth"
)
cfg.DEFAULT_CHECKPOINT_DIR = os.path.join(cfg.BASE_DIR, "checkpoints")

# Placeholder for loaded model points (set by utils.model_points.load_or_sample_model_points)
cfg.MODEL_POINTS_BY_ID = {}


# Lazy helper to load model points for a specific object and cache into cfg.MODEL_POINTS_BY_ID
def load_model_points_for_object(object_id=None, num_points=None):
    from utils.model_points import load_or_sample_model_points

    num = num_points if num_points is not None else cfg.train.num_model_points
    pts = load_or_sample_model_points(num, object_id=object_id)
    # cache by id (use 'default' for None)
    key = object_id if object_id is not None else "default"
    cfg.MODEL_POINTS_BY_ID[key] = pts
    return pts


def get_cached_model_points(object_id=None):
    key = object_id if object_id is not None else "default"
    return cfg.MODEL_POINTS_BY_ID.get(key, None)


cfg.load_model_points_for_object = load_model_points_for_object
cfg.get_cached_model_points = get_cached_model_points

# cap for dense loss sampling/chunking
cfg.dense_loss_max_pixels = 16384

# Automatically run prepare_model_points.py when a placeholder model_points.npy is detected and a mesh exists
cfg.auto_prepare_model_points = True
