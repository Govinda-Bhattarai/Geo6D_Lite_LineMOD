import torch
import os

def save_checkpoint(model, optimizer, epoch, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, f"epoch_{epoch+1}.pth")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")

def load_checkpoint(model, optimizer, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]
