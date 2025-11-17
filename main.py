import argparse
import torch
import os
import json
from tqdm import tqdm
from datetime import datetime

from config import cfg
from dataset import LineMODDriveMini
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from models.losses import (
    geodesic_loss,
    trans_l1_loss,
    reprojection_loss,
    dense_rot_loss,
    dense_trans_loss,
    conf_bce_loss,
)
from utils.checkpoint import save_checkpoint
from utils.metrics import rotation_error, translation_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(args):
    """Train the Geo6D model from scratch or resume."""
    print("🚀 Starting training mode...")
    # Use config-based paths via object_ids
    # Object IDs correspond to folders in datasets/Linemod_preprocessed/data/
    # e.g., "05", "09", "13" etc.
    object_ids = args.object_ids if args.object_ids else ["05"]  # Default to object 05

    # Handle max_samples argument
    # None/omit: use all frames (ignore split file)
    # 0: use split file (curated frames, typically 180)
    # >0: use that many samples (from split file if available, otherwise all frames)
    if args.max_samples is None:
        max_samples = None
        print(
            "📊 Using ALL available samples per object for training (ignoring split file)"
        )
    elif args.max_samples == 0:
        # Special case: 0 means use split file (curated frames) without limit
        max_samples = None  # Don't limit, but will use split file
        print(
            "📊 Using frames from split file (curated training set, typically 180 frames)"
        )
    else:
        max_samples = args.max_samples
        print(f"📊 Using max {max_samples} samples per object for training")

    dataset = LineMODDriveMini(
        object_ids=object_ids,
        split="train",
        max_per_obj=max_samples,
        use_augmentation=getattr(cfg.train, "use_augmentation", False),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True
    )

    backbone = ResNetBackbone(
        pretrained=cfg.model.pretrained,
        out_channels=cfg.model.feat_dim,
        backbone_type=cfg.model.backbone,  # Pass ResNet type (resnet18 or resnet34)
    )
    geo_channels = getattr(cfg.model, "geo_channels", 12)
    model = Geo6DNet(backbone, geo_channels=geo_channels).to(device)

    # Pose head now has its own improved initialization (see models/pose_head.py)
    print(
        f"✅ Using {cfg.model.backbone.upper()} backbone with {cfg.model.feat_dim}D features and improved residual pose head"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd
    )

    # Learning rate scheduler with warmup
    scheduler = None
    warmup_scheduler = None
    lr_was_reset = False  # Track if LR was reset on resume

    if hasattr(cfg.train, "lr_scheduler"):
        if cfg.train.lr_scheduler == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=getattr(cfg.train, "lr_factor", 0.5),
                patience=getattr(cfg.train, "lr_patience", 5),
                min_lr=getattr(cfg.train, "lr_min", 1e-6),
                verbose=True,
            )
            print(
                f"📉 Using ReduceLROnPlateau scheduler (patience={cfg.train.lr_patience}, factor={cfg.train.lr_factor})"
            )
        elif cfg.train.lr_scheduler == "step":
            from torch.optim.lr_scheduler import StepLR, LambdaLR

            # Warmup scheduler
            warmup_epochs = getattr(cfg.train, "lr_warmup_epochs", 0)
            warmup_factor = getattr(cfg.train, "lr_warmup_factor", 0.3)
            if warmup_epochs > 0:

                def warmup_lambda(epoch):
                    if epoch < warmup_epochs:
                        return warmup_factor + (1.0 - warmup_factor) * (
                            epoch / warmup_epochs
                        )
                    return 1.0

                warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

            # Step LR scheduler
            step_size = getattr(cfg.train, "lr_step_size", 10)
            gamma = getattr(cfg.train, "lr_gamma", 0.5)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            print(
                f"📉 Using StepLR scheduler (step_size={step_size}, gamma={gamma}) with {warmup_epochs} epoch warmup"
            )
        elif cfg.train.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

            # Warmup scheduler
            warmup_epochs = getattr(cfg.train, "lr_warmup_epochs", 0)
            warmup_factor = getattr(cfg.train, "lr_warmup_factor", 0.1)
            if warmup_epochs > 0:

                def warmup_lambda(epoch):
                    if epoch < warmup_epochs:
                        return warmup_factor + (1.0 - warmup_factor) * (
                            epoch / warmup_epochs
                        )
                    return 1.0

                warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

            # Cosine annealing scheduler
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg.train.num_epochs - warmup_epochs,
                eta_min=getattr(cfg.train, "lr_min", 1e-6),
            )
            print(
                f"📉 Using CosineAnnealingLR scheduler with {warmup_epochs} epoch warmup"
            )

    # Load per-object model points and cache
    for oid in object_ids:
        try:
            pts = cfg.load_model_points_for_object(
                oid, num_points=cfg.train.num_model_points
            )
            print(f"📐 Loaded model points for {oid}: {pts.shape}")
        except Exception as e:
            print(f"⚠️ Could not load model points for {oid}: {e}")
            pts = None

    start_epoch = 0
    resume_ckpt = args.resume_checkpoint if args.resume_checkpoint else args.checkpoint
    if args.resume and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]

        # Reset learning rate if it's too low (AGGRESSIVE FIX for fast convergence to 0.5)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < 3e-5:  # If LR has decayed too much
            # Use aggressive reset: 50% of initial LR for fast learning
            new_lr = cfg.train.lr * 0.5  # Use 50% of initial LR (should be ~6e-5)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            lr_was_reset = True
            print(f"🔁 Resumed training from epoch {start_epoch}")
            print(
                f"📈 RESET learning rate from {current_lr:.2e} to {new_lr:.2e} (aggressive reset for fast convergence)"
            )
            print("   Target: Loss < 0.5 by epoch 40")
        else:
            print(
                f"🔁 Resumed training from epoch {start_epoch} (LR: {current_lr:.2e})"
            )

        # Recreate scheduler if LR was reset (don't use old scheduler state)
        if lr_was_reset and scheduler is not None:
            # Recreate scheduler with adjusted T_max for remaining epochs
            warmup_epochs = getattr(cfg.train, "lr_warmup_epochs", 0)
            remaining_epochs = cfg.train.num_epochs - start_epoch
            if cfg.train.lr_scheduler == "cosine":
                from torch.optim.lr_scheduler import CosineAnnealingLR

                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=remaining_epochs,  # Only remaining epochs
                    eta_min=getattr(cfg.train, "lr_min", 5e-6),  # Use config's lr_min
                )
                print(
                    f"📉 Recreated CosineAnnealingLR scheduler for remaining {remaining_epochs} epochs"
                )
                print(
                    f"   LR will decay from {optimizer.param_groups[0]['lr']:.2e} to {cfg.train.lr_min:.2e}"
                )
        elif scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                # Only load scheduler state if LR wasn't reset
                if not lr_was_reset:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                print(
                    "⚠️  Could not load scheduler state, will continue with new scheduler"
                )

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_log.json")
    log_data = []

    for epoch in range(start_epoch, cfg.train.num_epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(
            loader, desc=f"Epoch [{epoch + 1}/{cfg.train.num_epochs}]", ncols=100
        )
        for batch_idx, batch in enumerate(progress):
            img = batch["img"].to(device)
            depth = (
                batch["depth"].to(device)
                if cfg.model.use_depth and batch["depth"] is not None
                else None
            )
            K = batch["K"].to(device)
            R_gt, t_gt = batch["R"].to(device), batch["t"].to(device)
            # Pass GT mask to model for proper confidence-weighted pooling (CRITICAL FIX)
            mask_for_model = batch.get("mask", None)
            if mask_for_model is not None:
                mask_for_model = mask_for_model.to(device)

            preds = model(img, depth, K, mask=mask_for_model)
            lossR = geodesic_loss(preds["R"], R_gt)
            lossT = trans_l1_loss(preds["t_off"], t_gt)
            # CRITICAL FIX: Apply loss weights from config
            loss = cfg.loss.w_rot * lossR + cfg.loss.w_trans * lossT

            # Reprojection loss (if model points and depth available)
            if cfg.loss.use_reproj and depth is not None and depth.sum() > 1e-6:
                # lookup object_id from batch (assume single-object or per-sample consistent)
                obj_id = None
                if "object_id" in batch:
                    # batch['object_id'] might be list/tuple of strings
                    v = batch["object_id"]
                    try:
                        # take first id
                        obj_id = v[0] if isinstance(v, (list, tuple)) else str(v)
                    except Exception:
                        obj_id = str(v)
                try:
                    mp = (
                        cfg.get_cached_model_points(obj_id)
                        if hasattr(cfg, "get_cached_model_points")
                        else None
                    )
                    if mp is None:
                        mp = cfg.load_model_points_for_object(
                            obj_id, num_points=cfg.train.num_model_points
                        )
                    mp_dev = (
                        mp.to(device)
                        if isinstance(mp, torch.Tensor)
                        else torch.tensor(mp, dtype=torch.float32, device=device)
                    )
                    reproj = reprojection_loss(
                        mp_dev,
                        preds["R"],
                        preds["t_off"],
                        K,
                        depth,
                        mask=batch.get("mask", None),
                    )
                    w_reproj = getattr(cfg.loss, "w_reproj", 1.0)
                    loss = loss + w_reproj * reproj
                except Exception:
                    import traceback

                    traceback.print_exc()
                    # continue without reprojection
                    pass

            # Dense supervision if head provides dense maps
            if "rot6d_map" in preds and "trans_map" in preds and "conf_map" in preds:
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(device)
                try:
                    dense_rot = dense_rot_loss(
                        preds["rot6d_map"], preds["R"], preds["conf_map"], mask=mask
                    )
                    dense_trans = dense_trans_loss(
                        preds["trans_map"], preds["t_off"], preds["conf_map"], mask=mask
                    )
                    conf_loss = conf_bce_loss(preds["conf_map"], mask)
                    loss = (
                        loss
                        + cfg.loss.w_dense_rot * dense_rot
                        + cfg.loss.w_dense_trans * dense_trans
                        + cfg.loss.w_conf * conf_loss
                    )
                except Exception:
                    import traceback

                    traceback.print_exc()
                    # continue without dense losses
                    pass

            optimizer.zero_grad()
            loss.backward()

            # CRITICAL FIX: Gradient clipping for stability (reduced to prevent explosion)
            if hasattr(cfg.train, "gradient_clip") and cfg.train.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.gradient_clip
                )
                # Log gradient norm occasionally for debugging
                if batch_idx % 50 == 0:
                    progress.set_postfix(
                        {"loss": f"{loss.item():.4f}", "grad": f"{grad_norm:.2f}"}
                    )

            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(loader)

        # Update learning rate scheduler
        warmup_epochs = getattr(cfg.train, "lr_warmup_epochs", 0)
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
            # Don't step main scheduler during warmup
        elif scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                # For step/cosine schedulers, step after warmup
                if epoch >= warmup_epochs or lr_was_reset:
                    # Step the scheduler
                    scheduler.step()
                    # If LR is too low and loss is still high, boost it to prevent stagnation
                    current_lr = optimizer.param_groups[0]["lr"]
                    # Boost LR if it's too low and loss is plateauing (only for step scheduler)
                    if (
                        isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
                        and current_lr < 4e-5
                        and avg_loss > 2.0
                    ):
                        # Boost LR moderately to continue learning
                        new_lr = min(
                            current_lr * 1.5, cfg.train.lr * 0.7
                        )  # Cap at 70% of initial LR, 1.5x boost
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = new_lr
                        print(
                            f"  📈 Boosted LR from {current_lr:.2e} to {new_lr:.2e} (loss plateauing, continuing learning)"
                        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{cfg.train.num_epochs}] | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
        )

        save_checkpoint(
            model, optimizer, epoch, path=args.save_dir, scheduler=scheduler
        )

        # log metrics
        log_data.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "timestamp": datetime.now().isoformat(),
            }
        )
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=4)

    print(f"🏁 Training complete! Logs saved at: {log_path}")


def evaluate_model(args):
    """Evaluate the model on the LineMOD dataset."""
    print("🧪 Starting evaluation mode...")
    # Use config-based paths via object_ids
    object_ids = args.object_ids if args.object_ids else ["05"]  # Default to object 05
    if args.max_samples is None:
        max_samples = None
        print("📊 Using ALL available samples per object for evaluation")
    else:
        max_samples = args.max_samples
        print(f"📊 Using max {max_samples} samples per object for evaluation")
    dataset = LineMODDriveMini(
        object_ids=object_ids, split="test", max_per_obj=max_samples
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    backbone = ResNetBackbone(pretrained=False)
    model = Geo6DNet(backbone).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"✅ Loaded checkpoint from {args.checkpoint}")
    else:
        print("⚠️ No checkpoint found — evaluating untrained model.")

    model.eval()
    rot_errs, trans_errs = [], []
    progress = tqdm(loader, desc="Evaluating", ncols=100)

    with torch.no_grad():
        for batch in progress:
            img = batch["img"].to(device)
            depth = (
                batch["depth"].to(device)
                if cfg.model.use_depth and batch["depth"] is not None
                else None
            )
            K = batch["K"].to(device)
            R_gt, t_gt = batch["R"].cpu().numpy(), batch["t"].cpu().numpy()
            # Pass GT mask to model for proper confidence-weighted pooling (CRITICAL FIX)
            mask_for_model = batch.get("mask", None)
            if mask_for_model is not None:
                mask_for_model = mask_for_model.to(device)
            preds = model(img, depth, K, mask=mask_for_model)
            R_pred = preds["R"].cpu().numpy()
            t_pred = preds["t_off"].cpu().numpy()

            for i in range(len(R_gt)):
                rot_err = rotation_error(R_gt[i], R_pred[i])
                trans_err = translation_error(t_gt[i], t_pred[i])
                rot_errs.append(rot_err)
                trans_errs.append(trans_err)
                progress.set_postfix(
                    {"RotErr": f"{rot_err:.2f}", "TransErr": f"{trans_err:.2f}"}
                )

    mean_rot = float(torch.tensor(rot_errs).mean())
    mean_trans = float(torch.tensor(trans_errs).mean())

    # Compute accuracy metrics with thresholds: 10° for rotation, 10 cm for translation
    from utils.metrics import compute_accuracy_metrics

    accuracy_metrics = compute_accuracy_metrics(
        rot_errs, trans_errs, rot_threshold=10.0, trans_threshold=10.0
    )

    print(f"\n📊 Mean Rotation Error: {mean_rot:.2f}°")
    print(f"📊 Mean Translation Error: {mean_trans:.2f} cm")
    print("\n✅ Accuracy Metrics (Threshold: 10° rotation, 10 cm translation):")
    print(
        f"   Rotation Accuracy: {accuracy_metrics['rotation_accuracy']:.2f}% ({accuracy_metrics['rotation_correct']}/{accuracy_metrics['total_samples']})"
    )
    print(
        f"   Translation Accuracy: {accuracy_metrics['translation_accuracy']:.2f}% ({accuracy_metrics['translation_correct']}/{accuracy_metrics['total_samples']})"
    )
    print(
        f"   Overall Accuracy: {accuracy_metrics['overall_accuracy']:.2f}% ({accuracy_metrics['overall_correct']}/{accuracy_metrics['total_samples']})"
    )

    results = {
        "mean_rotation_error": mean_rot,
        "mean_translation_error": mean_trans,
        "num_samples": len(rot_errs),
        "accuracy_metrics": accuracy_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    result_path = os.path.join(args.save_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Evaluation results saved at {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Geo6D-Lite LineMOD Trainer/Evaluator")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval",
    )
    parser.add_argument(
        "--object_ids",
        type=str,
        nargs="+",
        default=None,
        help="Object IDs to use (e.g., '05' '09'). Uses config-based paths.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        nargs="?",
        const=None,
        help="Max samples per object. Omit to use ALL frames (1196). Set to 200 to use split file (180 frames).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="[Legacy] Path to LineMOD dataset root (ignored if object_ids provided)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=cfg.DEFAULT_CHECKPOINT, help="Checkpoint path"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=cfg.DEFAULT_CHECKPOINT_DIR,
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume training. Optionally pass a checkpoint path.",
    )
    args = parser.parse_args()

    resume_flag = False
    resume_path = None
    if args.resume is True:
        resume_flag = True
        resume_path = args.checkpoint
    elif isinstance(args.resume, str):
        resume_flag = True
        resume_path = args.resume
    args.resume = resume_flag
    args.resume_checkpoint = resume_path

    if args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        evaluate_model(args)
    else:
        raise ValueError("Invalid mode. Use --mode train or --mode eval")


if __name__ == "__main__":
    main()
