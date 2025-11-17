#!/usr/bin/env python3
import argparse
import os
import sys
import torch

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

"""Small training script to run a few batches for debug.

Usage:
    python tools/run_small_train.py --object_id 05 --batches 5

This will initialize the model, run a few forward/backward steps on a tiny subset
and print loss components to verify reprojection/dense losses are functional.
"""

# Ensure repo root is on PYTHONPATH and make it the working directory so local imports resolve
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
try:
    os.chdir(repo_dir)
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_id", type=str, default="05")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    ds = LineMODDriveMini(object_ids=[args.object_id], split="train", max_per_obj=8)
    if len(ds) == 0:
        print("No samples found for object", args.object_id)
        return 2

    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    backbone = ResNetBackbone(pretrained=False)
    model = Geo6DNet(backbone).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd
    )

    # load model points for object
    mp = cfg.load_model_points_for_object(
        args.object_id, num_points=cfg.train.num_model_points
    )
    if mp is None:
        print("No model points available; reprojection loss will be skipped")
    else:
        mp = mp.to(device)

    it = iter(dl)
    for i in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        img = batch["img"].to(device)
        depth = (
            batch["depth"].to(device)
            if cfg.model.use_depth and batch["depth"] is not None
            else None
        )
        K = batch["K"].to(device)
        R_gt = batch["R"].to(device)
        t_gt = batch["t"].to(device)

        model.train()
        preds = model(img, depth, K)

        lossR = geodesic_loss(preds["R"], R_gt)
        lossT = trans_l1_loss(preds["t_off"], t_gt)
        total = lossR + lossT

        if (
            cfg.loss.use_reproj
            and depth is not None
            and depth.sum() > 1e-6
            and mp is not None
        ):
            try:
                reproj = reprojection_loss(
                    mp,
                    preds["R"],
                    preds["t_off"],
                    K,
                    depth,
                    mask=batch.get("mask", None),
                )
                print(f"Batch {i}: reproj={reproj.item():.6f}")
                total = total + getattr(cfg.loss, "w_reproj", 1.0) * reproj
            except Exception as e:
                print("Reproj error:", e)

        if "rot6d_map" in preds:
            try:
                dense_rot = dense_rot_loss(
                    preds["rot6d_map"],
                    preds["R"],
                    preds["conf_map"],
                    mask=batch.get("mask", None),
                )
                dense_trans = dense_trans_loss(
                    preds["trans_map"],
                    preds["t_off"],
                    preds["conf_map"],
                    mask=batch.get("mask", None),
                )
                conf_loss = conf_bce_loss(preds["conf_map"], batch.get("mask", None))
                print(
                    f"Batch {i}: dense_rot={dense_rot.item():.6f} dense_trans={dense_trans.item():.6f} conf={conf_loss.item():.6f}"
                )
                total = (
                    total
                    + cfg.loss.w_dense_rot * dense_rot
                    + cfg.loss.w_dense_trans * dense_trans
                    + cfg.loss.w_conf * conf_loss
                )
            except Exception as e:
                print("Dense loss error:", e)

        opt.zero_grad()
        total.backward()
        opt.step()

        print(
            f"Batch {i}: lossR={lossR.item():.6f} lossT={lossT.item():.6f} total={total.item():.6f}"
        )

    print("Done small training run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
