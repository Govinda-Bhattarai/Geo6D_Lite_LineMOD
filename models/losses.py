import torch
import torch.nn.functional as F

def geodesic_loss(Rp, Rg):
    Rrel = torch.matmul(Rp.transpose(1, 2), Rg)
    cos = (torch.diagonal(Rrel, dim1=1, dim2=2).sum(1) - 1) * 0.5
    cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()

def trans_l1_loss(tp, tg):
    return F.l1_loss(tp, tg)
