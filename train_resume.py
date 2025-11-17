import os
import torch
import types

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from models.losses import geodesic_loss, trans_l1_loss
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = cfg.DEFAULT_CHECKPOINT
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

torch.serialization.add_safe_globals([types.SimpleNamespace])


def build_model():
    backbone = ResNetBackbone(pretrained=cfg.model.pretrained)
    model = Geo6DNet(backbone).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs, eta_min=1e-6)
    return model, optimizer, scheduler


def resume_if_exists(model, optimizer, scheduler):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"🔁 Resumed from checkpoint (epoch {start_epoch})")
        return start_epoch
    else:
        print("🚀 Starting fresh (no checkpoint found)")
        return 0


def train_one_step(batch, model, optimizer):
    model.train()
    img = batch["img"].to(device)
    depth = batch["depth"].to(device) if cfg.model.use_depth else None
    K = batch["K"].to(device)
    R_gt, t_gt = batch["R"].to(device), batch["t"].to(device)

    out = model(img, depth, K)
    lossR = geodesic_loss(out["R"], R_gt)
    lossT = trans_l1_loss(out["t_off"], t_gt)
    loss = lossR + lossT

    # Reprojection loss: requires model points and depth
    try:
        from models.losses import reprojection_loss

        # Get model points — try to load from cfg if provided
        # avoid using placeholder zero depth
        if depth is not None and depth.sum() > 1e-6:
            # try to use cached per-object model points (if main loaded per-object)
            obj_id = batch.get("object_id", None) if isinstance(batch, dict) else None
            mp = (
                cfg.get_cached_model_points(obj_id)
                if hasattr(cfg, "get_cached_model_points")
                else None
            )
            if mp is None:
                # fallback: lazy-load default
                try:
                    mp = cfg.load_model_points_for_object(obj_id)
                except Exception:
                    mp = None
            if mp is None:
                P = (
                    cfg.train.num_model_points
                    if hasattr(cfg.train, "num_model_points")
                    else 500
                )
                mp = torch.rand(P, 3, device=img.device) - 0.5
            # ensure mp on correct device
            if isinstance(mp, torch.Tensor):
                mp_dev = mp.to(img.device)
            else:
                mp_dev = torch.tensor(mp, dtype=torch.float32, device=img.device)
            reproj = reprojection_loss(
                mp_dev, out["R"], out["t_off"], K, depth, mask=batch.get("mask", None)
            )
            # weight reprojection loss if config has w_reproj
            w_reproj = getattr(cfg.loss, "w_reproj", 1.0)
            loss = loss + w_reproj * reproj
    except Exception:
        # if reprojection fails for any reason, skip it (warn in logs could be added)
        pass

    # Dense supervision (if model provides dense maps)
    if "rot6d_map" in out and "trans_map" in out and "conf_map" in out:
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)
        dense_rot = __import__(
            "models.losses", fromlist=["dense_rot_loss"]
        ).dense_rot_loss(out["rot6d_map"], out["R"], out["conf_map"], mask=mask)
        dense_trans = __import__(
            "models.losses", fromlist=["dense_trans_loss"]
        ).dense_trans_loss(out["trans_map"], out["t_off"], out["conf_map"], mask=mask)
        conf_loss = __import__(
            "models.losses", fromlist=["conf_bce_loss"]
        ).conf_bce_loss(out["conf_map"], mask)

        # Weight dense losses using config
        loss = (
            loss
            + cfg.loss.w_dense_rot * dense_rot
            + cfg.loss.w_dense_trans * dense_trans
            + cfg.loss.w_conf * conf_loss
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    model, optimizer, scheduler = build_model()
    start_epoch = resume_if_exists(model, optimizer, scheduler)
    print(f"Model loaded. Starting at epoch {start_epoch}.")
