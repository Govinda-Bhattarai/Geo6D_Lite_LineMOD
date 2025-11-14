import torch
from torch.utils.data import DataLoader
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from models.losses import geodesic_loss, trans_l1_loss
from utils.checkpoint import save_checkpoint
from config import cfg

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(train_loader):
    backbone = ResNetBackbone(pretrained=cfg.model.pretrained)
    model = Geo6DNet(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)

    for epoch in range(cfg.train.num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            imgs, depths, Ks, R_gt, t_gt = batch
            imgs, depths, Ks = imgs.to(device), depths.to(device), Ks.to(device)
            R_gt, t_gt = R_gt.to(device), t_gt.to(device)

            preds = model(imgs, depths, Ks)
            loss_R = geodesic_loss(preds["R"], R_gt)
            loss_t = trans_l1_loss(preds["t_off"], t_gt)
            loss = loss_R + loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{cfg.train.num_epochs}] | Loss: {total_loss:.4f}")
        save_checkpoint(model, optimizer, epoch)

if __name__ == "__main__":
    train_loader = []  # Replace with real DataLoader
    train(train_loader)
