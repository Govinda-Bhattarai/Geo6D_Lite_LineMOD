import torch
import torch.nn as nn
import torch.optim as optim
import os, copy
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from utils.checkpoint import save_checkpoint

# ==========================================
# 🔧 Helper Functions
# ==========================================
def build_model_optimizer(device):
    backbone = ResNetBackbone(pretrained=False)
    model = Geo6DNet(backbone).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer

def dummy_data(device):
    x = torch.randn(2, 3, 224, 224, device=device)
    d = torch.randn(2, 1, 224, 224, device=device)
    K = torch.eye(3, device=device).unsqueeze(0).repeat(2, 1, 1)
    target_R = torch.randn(2, 3, 3, device=device)
    target_t = torch.randn(2, 3, device=device)
    target_rot6d = torch.randn(2, 6, device=device)
    return x, d, K, target_R, target_t, target_rot6d

def compute_loss(model, x, d, K, target_R, target_t, target_rot6d):
    criterion = nn.MSELoss()
    out = model(x, d, K)
    return (
        criterion(out["R"], target_R)
        + criterion(out["t_off"], target_t)
        + criterion(out["rot6d"], target_rot6d)
    )

# ==========================================
# 🧪 Tests
# ==========================================
def test_checkpoint_pipeline(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Forward & Save ---
    model, optimizer = build_model_optimizer(device)
    x, d, K, target_R, target_t, target_rot6d = dummy_data(device)
    loss = compute_loss(model, x, d, K, target_R, target_t, target_rot6d)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    ckpt_path = tmp_path / "epoch_0.pth"
    save_checkpoint(model, optimizer, epoch=0, path=tmp_path)
    assert ckpt_path.exists(), "Checkpoint was not saved!"

    # --- Step 2: Reload & Compare Weights ---
    model2, optimizer2 = build_model_optimizer(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt["model_state_dict"])
    optimizer2.load_state_dict(ckpt["optimizer_state_dict"])

    checksum1 = sum([p.abs().sum().item() for p in model.state_dict().values()])
    checksum2 = sum([p.abs().sum().item() for p in model2.state_dict().values()])
    assert abs(checksum1 - checksum2) < 1e-8, "Model weights differ after reload!"

    # --- Step 3: Resume & Continue Training ---
    loss_before = compute_loss(model2, x, d, K, target_R, target_t, target_rot6d).item()
    model2.train()
    for _ in range(2):
        loss = compute_loss(model2, x, d, K, target_R, target_t, target_rot6d)
        optimizer2.zero_grad(); loss.backward(); optimizer2.step()
    loss_after = compute_loss(model2, x, d, K, target_R, target_t, target_rot6d).item()
    assert loss_after <= loss_before + 1e-6, "Loss did not decrease after resuming training!"

    # --- Step 4: Numerical Reproducibility ---
    torch.manual_seed(42)
    optimizer2.zero_grad()
    loss_ref = compute_loss(model2, x, d, K, target_R, target_t, target_rot6d)
    loss_ref.backward()
    grads_ref = [p.grad.clone() for p in model2.parameters() if p.grad is not None]

    # Save & reload again
    torch.save({
        "model_state_dict": model2.state_dict(),
        "optimizer_state_dict": optimizer2.state_dict()
    }, tmp_path / "repro_test.pth")

    model3, optimizer3 = build_model_optimizer(device)
    ckpt = torch.load(tmp_path / "repro_test.pth", map_location=device, weights_only=False)
    model3.load_state_dict(ckpt["model_state_dict"])
    optimizer3.load_state_dict(ckpt["optimizer_state_dict"])

    optimizer3.zero_grad()
    loss_new = compute_loss(model3, x, d, K, target_R, target_t, target_rot6d)
    loss_new.backward()
    grads_new = [p.grad.clone() for p in model3.parameters() if p.grad is not None]

    grad_diff = sum([(g1 - g2).abs().sum().item() for g1, g2 in zip(grads_ref, grads_new)])
    assert grad_diff < 1e-9, f"Gradients differ after reload: {grad_diff:.2e}"

    print("✅ Checkpoint pipeline verified: save/load/resume/reproducibility all passed.")

