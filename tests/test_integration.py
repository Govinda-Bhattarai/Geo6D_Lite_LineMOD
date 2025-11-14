import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
import torch

def test_model_forward():
    backbone = ResNetBackbone(pretrained=False)
    model = Geo6DNet(backbone)
    x = torch.randn(1, 3, 224, 224)
    d = torch.randn(1, 1, 224, 224)
    K = torch.eye(3).unsqueeze(0)
    out = model(x, d, K)
    assert "R" in out and "t_off" in out
