import torch
import torch.nn.functional as F

def rot6d_to_matrix(x):
    a1, a2 = x[:, 0:3], x[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)
