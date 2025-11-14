from types import SimpleNamespace as NS

cfg = NS(
    model=NS(
        backbone="resnet18",
        pretrained=True,
        use_depth=True,
        use_mask=True,
        feat_dim=256,
        rot_repr="ortho6d",
        input_res=256,
    ),
    loss=NS(
        w_rot=1.0,
        w_trans=1.0,
        w_reproj=1.0,
    ),
    train=NS(
        lr=1e-4,
        wd=1e-4,
        batch_size=4,
        num_model_points=500,
        num_epochs=12,
    ),
)
