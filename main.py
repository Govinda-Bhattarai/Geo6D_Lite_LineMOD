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
from models.losses import geodesic_loss, trans_l1_loss
from utils.checkpoint import save_checkpoint
from utils.metrics import rotation_error, translation_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(args):
    """Train the Geo6D model from scratch or resume."""
    print("🚀 Starting training mode...")
    data_root = args.data_root or "/content/drive/MyDrive/SharedCheckpoints/datasets/LineMOD/lm"
    objects = ["ape", "can", "driller"]

    dataset = LineMODDriveMini(data_root, objects, max_per_obj=200)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    backbone = ResNetBackbone(pretrained=cfg.model.pretrained)
    model = Geo6DNet(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)

    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"🔁 Resumed training from epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_log.json")
    log_data = []

    for epoch in range(start_epoch, cfg.train.num_epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch [{epoch+1}/{cfg.train.num_epochs}]", ncols=100)
        for batch in progress:
            img = batch["img"].to(device)
            depth = batch["depth"].to(device) if cfg.model.use_depth and batch["depth"] is not None else None
            K = batch["K"].to(device)
            R_gt, t_gt = batch["R"].to(device), batch["t"].to(device)

            preds = model(img, depth, K)
            lossR = geodesic_loss(preds["R"], R_gt)
            lossT = trans_l1_loss(preds["t_off"], t_gt)
            loss = lossR + lossT

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(loader)
        print(f"✅ Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, path=args.save_dir)

        # log metrics
        log_data.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "timestamp": datetime.now().isoformat()
        })
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=4)

    print(f"🏁 Training complete! Logs saved at: {log_path}")


def evaluate_model(args):
    """Evaluate the model on the LineMOD dataset."""
    print("🧪 Starting evaluation mode...")
    data_root = args.data_root or "/content/drive/MyDrive/SharedCheckpoints/datasets/LineMOD/lm"
    objects = ["ape"]
    dataset = LineMODDriveMini(data_root, objects, max_per_obj=50)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    backbone = ResNetBackbone(pretrained=False)
    model = Geo6DNet(backbone).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
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
            depth = batch["depth"].to(device) if cfg.model.use_depth and batch["depth"] is not None else None
            K = batch["K"].to(device)
            R_gt, t_gt = batch["R"].cpu().numpy(), batch["t"].cpu().numpy()
            preds = model(img, depth, K)
            R_pred = preds["R"].cpu().numpy()
            t_pred = preds["t_off"].cpu().numpy()

            for i in range(len(R_gt)):
                rot_err = rotation_error(R_gt[i], R_pred[i])
                trans_err = translation_error(t_gt[i], t_pred[i])
                rot_errs.append(rot_err)
                trans_errs.append(trans_err)
                progress.set_postfix({"RotErr": f"{rot_err:.2f}", "TransErr": f"{trans_err:.2f}"})

    mean_rot = float(torch.tensor(rot_errs).mean())
    mean_trans = float(torch.tensor(trans_errs).mean())

    print(f"\n📊 Mean Rotation Error: {mean_rot:.2f}°")
    print(f"📊 Mean Translation Error: {mean_trans:.2f} cm")

    results = {
        "mean_rotation_error": mean_rot,
        "mean_translation_error": mean_trans,
        "num_samples": len(rot_errs),
        "timestamp": datetime.now().isoformat()
    }
    result_path = os.path.join(args.save_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Evaluation results saved at {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Geo6D-Lite LineMOD Trainer/Evaluator")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--data_root", type=str, default=None, help="Path to LineMOD dataset root")
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/SharedCheckpoints/geo6d_checkpoints/geo6d_lite_latest.pth", help="Checkpoint path")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/SharedCheckpoints/geo6d_checkpoints", help="Directory to save checkpoints and logs")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if available")
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        evaluate_model(args)
    else:
        raise ValueError("Invalid mode. Use --mode train or --mode eval")


if __name__ == "__main__":
    main()
