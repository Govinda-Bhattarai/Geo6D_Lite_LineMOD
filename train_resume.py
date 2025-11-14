import os, torch, types
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from models.losses import geodesic_loss, trans_l1_loss
from utils.checkpoint import save_checkpoint
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "/content/drive/MyDrive/SharedCheckpoints/geo6d_checkpoints/geo6d_lite_latest.pth"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

torch.serialization.add_safe_globals([types.SimpleNamespace])

def build_model():
    backbone = ResNetBackbone(pretrained=cfg.model.pretrained)
    model = Geo6DNet(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=cfg.train.wd)
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

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    model, optimizer, scheduler = build_model()
    start_epoch = resume_if_exists(model, optimizer, scheduler)
    print(f"Model loaded. Starting at epoch {start_epoch}.")
