import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, f"epoch_{epoch}.pth")

    # Handle both dicts and torch modules
    model_state = model if isinstance(model, dict) else model.state_dict()
    optim_state = optimizer if isinstance(optimizer, dict) else optimizer.state_dict()

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model_state,
        "optimizer_state_dict": optim_state,
    }, ckpt_path)
    print(f"✅ Saved checkpoint at {ckpt_path}")


def load_checkpoint(model, optimizer, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]
