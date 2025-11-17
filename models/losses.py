import torch
import torch.nn.functional as F
import numpy as np
from models.rot6d import rot6d_to_matrix


def geodesic_loss(Rp, Rg):
    Rrel = torch.matmul(Rp.transpose(1, 2), Rg)
    cos = (torch.diagonal(Rrel, dim1=1, dim2=2).sum(1) - 1) * 0.5
    cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


def trans_l1_loss(tp, tg):
    return F.l1_loss(tp, tg)


# ------------------------- Dense / reprojection-style losses -------------------------
def dense_rot_loss(rot6d_map, R_global, conf_map, mask=None):
    """
    Encourage per-pixel rotation predictions (rot6d_map) to agree with the global rotation R_global.
    rot6d_map: (B,6,H,W)
    R_global: (B,3,3)
    conf_map: (B,1,H,W) probabilities in [0,1]
    mask: optional binary mask (B,1,H,W) where foreground=1
    Returns average geodesic rotation error weighted by conf_map and mask.
    """
    # try to read sampling config
    try:
        from config import cfg

        max_pixels = getattr(cfg, "dense_loss_max_pixels", 16384)
    except Exception:
        max_pixels = 16384

    B, _, H, W = rot6d_map.shape
    N = H * W

    # If mask provided at image resolution, resize it to feature resolution
    if mask is not None:
        try:
            if mask.shape[-2:] != (H, W):
                mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        except Exception:
            mask = mask.float()

    # Flatten spatial dims
    rot_flat = rot6d_map.view(B, 6, -1).permute(0, 2, 1).contiguous()  # (B,N,6)
    weights = conf_map.view(B, -1)  # (B,N)
    if mask is not None:
        weights = weights * (mask.view(B, -1).float())

    # Decide whether to sample pixels
    if N > max_pixels:
        # sample indices per image according to weights (prob proportional to weight)
        inds = []
        for b in range(B):
            w = weights[b].cpu().numpy().astype(float)
            total = w.sum()
            if total <= 1e-6:
                # fallback uniform sampling
                idx = np.random.choice(N, max_pixels, replace=False)
            else:
                p = w / total
                idx = np.random.choice(N, max_pixels, replace=False, p=p)
            inds.append(idx)
        inds = np.stack(inds, 0)  # (B, max_pixels)

        angs = []
        for b in range(B):
            idx = inds[b]
            rot_b = rot_flat[b, idx, :].view(-1, 6)  # (S,6)
            R_pred_b = rot6d_to_matrix(rot_b)  # (S,3,3)
            Rg_b = R_global[b].unsqueeze(0).expand(R_pred_b.shape[0], -1, -1)
            Rrel_b = torch.matmul(R_pred_b.transpose(1, 2), Rg_b)
            cos_b = (torch.diagonal(Rrel_b, dim1=1, dim2=2).sum(1) - 1) * 0.5
            cos_b = torch.clamp(cos_b, -1 + 1e-6, 1 - 1e-6)
            ang_b = torch.acos(cos_b)
            w_b = torch.from_numpy(weights[b].cpu().numpy()[idx]).to(rot_b.device)
            denom = w_b.sum() + 1e-6
            per_image = (ang_b * w_b).sum() / denom
            angs.append(per_image)
        return torch.stack(angs).mean()

    # If small enough, compute on all pixels in chunks to limit memory
    rot_flat_b = rot_flat.view(-1, 6)  # (B*N,6)
    chunk = 16384
    angs = rot6d_map.new_zeros(B)
    for start in range(0, rot_flat_b.shape[0], chunk):
        end = min(rot_flat_b.shape[0], start + chunk)
        part = rot_flat_b[start:end, :]
        R_pred_p = rot6d_to_matrix(part)  # (M,3,3)
        m = R_pred_p.shape[0]
        # accumulate per-batch using slicing (simpler approach: rebuild full per-batch)
        # for clarity and safety, we rebuild per-batch arrays for this chunk
        for i in range(B):
            # indices in this chunk that belong to batch i
            s_idx = max(0, i * N - start)
            e_idx = min(m, (i + 1) * N - start)
            if s_idx >= e_idx:
                continue
            R_pred_i = R_pred_p[s_idx:e_idx]
            Rg_i = (
                R_global[i]
                .unsqueeze(0)
                .expand(R_pred_i.shape[0], -1, -1)
                .to(R_pred_i.device)
            )
            Rrel_i = torch.matmul(R_pred_i.transpose(1, 2), Rg_i)
            cos_i = (torch.diagonal(Rrel_i, dim1=1, dim2=2).sum(1) - 1) * 0.5
            cos_i = torch.clamp(cos_i, -1 + 1e-6, 1 - 1e-6)
            ang_i = torch.acos(cos_i)
            # weights for those pixels
            w_b = weights[i]
            # corresponding indices
            idxs = torch.arange(start, end, device=weights.device) - i * N
            idxs = idxs[(idxs >= 0) & (idxs < N)].long()
            if idxs.numel() == 0:
                continue
            w_sel = w_b[idxs].to(ang_i.device)
            denom = w_sel.sum() + 1e-6
            per_image = (ang_i * w_sel).sum() / denom
            angs[i] += per_image
    return angs.mean()


def dense_trans_loss(trans_map, t_global, conf_map, mask=None):
    """
    Encourage per-pixel translation predictions to agree with global translation.
    trans_map: (B,3,H,W)
    t_global: (B,3)
    conf_map: (B,1,H,W)
    mask: optional binary mask (B,1,H,W)
    Returns weighted L1 loss averaged over batch.
    """
    B, _, H, W = trans_map.shape
    N = H * W

    # Resize mask to feature resolution if provided
    if mask is not None:
        try:
            if mask.shape[-2:] != (H, W):
                mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        except Exception:
            mask = mask.float()

    trans_flat = trans_map.view(B, 3, -1).permute(0, 2, 1).contiguous()  # (B,N,3)
    t_exp = t_global.unsqueeze(1).expand(-1, N, -1)  # (B,N,3)

    per_pixel = F.l1_loss(trans_flat, t_exp, reduction="none").mean(-1)  # (B,N)

    weights = conf_map.view(B, -1)
    if mask is not None:
        weights = weights * (mask.view(B, -1).float())

    denom = weights.sum(1) + 1e-6
    per_image = (per_pixel * weights).sum(1) / denom
    return per_image.mean()


def conf_bce_loss(conf_map, mask):
    """
    Binary cross-entropy loss to encourage confidence map to match foreground mask.
    conf_map: (B,1,H,W) probabilities in [0,1]
    mask: (B,1,H,W) binary (0/1)
    """
    if mask is None:
        return torch.tensor(0.0, device=conf_map.device)
    # Resize mask if needed
    try:
        if mask.shape[-2:] != conf_map.shape[-2:]:
            mask = F.interpolate(mask.float(), size=conf_map.shape[-2:], mode="nearest")
        else:
            mask = mask.float()
    except Exception:
        mask = mask.float()
    return F.binary_cross_entropy(conf_map, mask)


def reprojection_loss(model_points, R, t, K, depth, mask=None):
    """
    Reprojection depth loss: project model_points (P,3) through predicted pose (R,t)
    and compare their z-values with ground-truth `depth` via bilinear sampling.

    Args:
        model_points: (P,3) or (B,P,3) in object coordinates (same units as t and depth)
        R: (B,3,3)
        t: (B,3)
        K: (B,3,3)
        depth: (B,1,H,W)
        mask: optional (B,1,H,W) foreground mask to weight points

    Returns:
        mean L1 difference between predicted point depths and sampled depth at projected pixels.
    """
    if depth is None:
        # Can't compute reprojection without depth
        return torch.tensor(0.0, device=R.device)

    B = R.shape[0]
    device = R.device

    # Prepare model points: ensure shape (B,P,3)
    if model_points.dim() == 2:
        P = model_points.shape[0]
        mp = model_points.unsqueeze(0).expand(B, -1, -1).to(device)
    elif model_points.dim() == 3:
        mp = model_points.to(device)
        P = mp.shape[1]
    else:
        raise ValueError("model_points must be (P,3) or (B,P,3)")

    # Transform model points to camera frame: X_cam = R @ X + t
    # mp: (B,P,3) -> permute to (B,3,P)
    X = mp.permute(0, 2, 1)  # (B,3,P)
    X_cam = torch.matmul(R, X) + t.unsqueeze(-1)  # (B,3,P)
    z = X_cam[:, 2, :].clamp(min=1e-6)  # (B,P)

    # Project to pixel coordinates
    fx = K[:, 0, 0].unsqueeze(-1)  # (B,1)
    fy = K[:, 1, 1].unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1)

    x = (X_cam[:, 0, :] / z) * fx + cx  # (B,P)
    y = (X_cam[:, 1, :] / z) * fy + cy  # (B,P)

    _, _, H, W = depth.shape

    # Normalize for grid_sample: x in [0,W-1] -> xn in [-1,1]
    xn = (x / (W - 1)) * 2.0 - 1.0
    yn = (y / (H - 1)) * 2.0 - 1.0

    # Build sampling grid (B, P, 1, 2) with (x,y) order
    grid = torch.stack([xn, yn], dim=-1).view(B, P, 1, 2)

    # Sample depth at projected locations
    # grid_sample expects grid in (B, Hout, Wout, 2) where last dim is (x, y)
    #    sampled = F.grid_sample(depth, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    # Use align_corners=True to match normalization using (W-1)/(H-1)
    sampled = F.grid_sample(
        depth, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    sampled = sampled.view(B, -1)  # (B,P)

    # Valid mask: within image bounds
    valid = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1))
    valid = valid.float()

    weights = valid
    if mask is not None:
        # sample mask values at projected points to prefer foreground
        mask_s = F.grid_sample(
            mask.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(B, -1)
        weights = weights * mask_s

    denom = weights.sum(1) + 1e-6

    per_point_error = (z - sampled).abs()  # (B,P)
    per_image = (per_point_error * weights).sum(1) / denom
    return per_image.mean()


# -------------------------------------------------------------------------------------
