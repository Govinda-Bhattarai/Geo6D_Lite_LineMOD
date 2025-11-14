import os, json, torch, numpy as np
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from dataset import LineMODDriveMini
from utils.metrics import rotation_error, translation_error
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader):
    model.eval()
    rot_errs, trans_errs = [], []
    with torch.no_grad():
        for batch in dataloader:
            imgs, depths, Ks = batch["img"].to(device), batch["depth"].to(device), batch["K"].to(device)
            R_gt, t_gt = batch["R"].cpu().numpy(), batch["t"].cpu().numpy()

            preds = model(imgs, depths, Ks)
            R_pred = preds["R"].cpu().numpy()
            t_pred = preds["t_off"].cpu().numpy()

            for i in range(len(R_gt)):
                rot_errs.append(rotation_error(R_gt[i], R_pred[i]))
                trans_errs.append(translation_error(t_gt[i], t_pred[i]))

    mean_rot = np.mean(rot_errs)
    mean_trans = np.mean(trans_errs)
    print(f"📊 Mean Rotation Error: {mean_rot:.2f}°")
    print(f"📊 Mean Translation Error: {mean_trans:.2f} cm")
    return mean_rot, mean_trans


if __name__ == "__main__":
    data_root = "/content/drive/MyDrive/SharedCheckpoints/datasets/LineMOD/lm"
    test_set = LineMODDriveMini(data_root, ["ape"], max_per_obj=10)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    backbone = ResNetBackbone(pretrained=False)
    model = Geo6DNet(backbone).to(device)

    ckpt_path = "/content/drive/MyDrive/SharedCheckpoints/geo6d_checkpoints/geo6d_lite_latest.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"✅ Loaded checkpoint from {ckpt_path}")
    else:
        print("⚠️ No checkpoint found. Using randomly initialized model.")

    evaluate(model, test_loader)
