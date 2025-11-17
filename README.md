
# Geo6D-Lite — 6D Pose Estimation on LineMOD

Geo6D-Lite is a research-friendly implementation of a modern 6D pose pipeline for LineMOD.  
It couples a multi-scale ResNet backbone with a Geo6D pose head, dense geometric features,
and production-ready inference scripts for evaluation, CLI use, and a REST API.

---

## Highlights
- **Architecture**: ResNet34 backbone with multi-scale fusion + residual Geo6D head (6 residual blocks, confidence-weighted pooling).
- **Geometric channels**: 12-channel tensor (pixel coords, normalized coords, normalized XYZ, centered XYZ, distance features).
- **Losses**: Geodesic rotation + L1 translation with optional dense supervision and reprojection.
- **Tooling**: Unified trainer (`main.py`), evaluation runner (`evaluate.py`), CLI inference (`infer.py`), API server (`api_server.py`), Dockerfile, and monitoring utilities in `tools/`.
- **Deployment**: Standalone inference wrapper (`inference.py`), FastAPI-compatible Flask server, Docker + docker-compose definitions.

---

## Repository Layout
```
Geo6D_Lite_LineMOD/
├── config.py                  # Hyperparameters & dataset helper
├── dataset.py                 # LineMOD loader (crop, intrinsics fix, augmentation)
├── main.py                    # Training & eval entry point
├── evaluate.py                # Standalone evaluation CLI
├── infer.py                   # Convenience wrapper around evaluate.py thresholds
├── inference.py               # Production PoseEstimator class (RGB+depth+K+mask)
├── api_server.py              # REST API using PoseEstimator
├── Dockerfile / docker-compose.yml
├── models/
│   ├── backbone.py            # ResNet18/34 with multi-scale fusion
│   ├── pose_head.py           # Residual Geo6D head + dense outputs
│   ├── losses.py              # Geodesic / L1 / reprojection / dense losses
│   └── rot6d.py               # 6D rotation utilities
├── utils/                     # Augmentation, checkpoints, metrics, model points
├── tools/                     # Monitoring & helper scripts
└── checkpoints/               # (git-ignored) training artifacts
```

---

## Installation
```bash
git clone https://github.com/Govinda-Bhattarai/Geo6D_Lite_LineMOD.git
cd Geo6D_Lite_LineMOD

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset
Download the preprocessed LineMOD split (BOP format) and place it under:
```
datasets/Linemod_preprocessed/data/<object_id>/
```
`dataset.py` automatically builds paths via `cfg.get_linemod_paths(object_id)` so the default structure works out of the box. Depth values must stay in meters (loader already converts from millimeters).

---

## Training & Evaluation
```bash
# Train from scratch on object 05
python main.py --mode train --object_ids 05

# Resume with optional checkpoint path (default is cfg.DEFAULT_CHECKPOINT)
python main.py --mode train --object_ids 05 --resume checkpoints/epoch_15.pth

# Evaluate a checkpoint
python main.py --mode eval --object_ids 05 --checkpoint checkpoints/epoch_35.pth
```

`main.py` handles:
- Warmup + StepLR / Cosine schedulers
- Gradient clipping and optional LR boosts
- Automatic model-point loading for reprojection
- Logging to `checkpoints/train_log.json` and per-epoch checkpoints

### Quick evaluation helper
```bash
# Sweeps thresholds (rotation in degrees, translation in cm)
python infer.py --object_ids 05 --checkpoint checkpoints/epoch_35.pth \
  --rotation_threshold 10 --translation_threshold 10
```

### Standalone evaluate script
```bash
python evaluate.py --object_ids 05 --checkpoint checkpoints/epoch_35.pth \
  --rot_thresh 10 --trans_thresh 0.10
```

---

## Inference & Deployment

### PoseEstimator class
```python
from inference import PoseEstimator
import numpy as np

estimator = PoseEstimator("checkpoints/epoch_35.pth", object_id="05")
rgb   = np.random.rand(256,256,3).astype("float32")
depth = np.ones((256,256), dtype="float32")
rot6d, trans, R = estimator.predict(rgb, depth)
```

### REST API
```bash
python api_server.py --checkpoint checkpoints/epoch_35.pth --object_id 05 --port 5000

curl -X POST http://localhost:5000/predict \
  -F image=@path/to/rgb.png \
  -F depth=@path/to/depth.png \
  -F mask=@path/to/mask.png \
  -F fx=572.4114 -F fy=573.5704 -F cx=325.2611 -F cy=242.0489
```
- Endpoints: `/health`, `/info`, `/predict`
- Requires RGB + depth; intrinsics optional (falls back to dataset defaults).

### Docker
```bash
docker build -t geo6d-lite .
docker run -p 5000:5000 geo6d-lite
# or
docker-compose up -d
```

---

## Tools & Utilities
- `verify_setup.py`: sanity-check imports, dataset paths, model forward pass.
- `tools/monitor_accuracy.py`: simple accuracy tracking over training.
- `tools/run_small_train.py`: short sanity training run.
- `visualize_dataset.py`, `visualize_results.py`: quick inspection scripts.

---

## Model At a Glance
| Component        | Details                                              |
|-----------------|------------------------------------------------------|
| Backbone        | ResNet34 (optional ResNet18) with multi-scale fusion |
| Pose head       | 6 residual blocks, confidence-weighted pooling       |
| Rotation        | Ortho6D → rotation matrix (Zhou et al. CVPR 2019)    |
| Translation     | Dense translation map + weighted pooling             |
| Losses          | Geodesic rot + L1 trans (+ optional reproj/dense)    |
| Optimizer       | AdamW + warmup + StepLR/Cosine                        |

Best checkpoints are stored under `checkpoints/` (ignored by git). Track your own metrics inside `FINAL_RESULTS.md` or similar, but strip sensitive data before publishing.

---

## Contributing / Customizing
- Adjust hyperparameters in `config.py` (feat_dim, loss weights, scheduler).
- Extend dataset loader for multi-object training by passing more IDs.
- Export ONNX / TensorRT by reusing the `PoseEstimator` forward method.

PRs that improve training stability, evaluation tooling, or deployment recipes are welcome.

---

## References
- LineMOD dataset — BOP Benchmark
- Ortho6D rotation representation — Zhou et al., CVPR 2019
- PyTorch documentation for model / optimizer APIs
