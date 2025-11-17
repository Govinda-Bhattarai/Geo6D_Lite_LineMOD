"""
REST API server for Geo6D-Lite pose estimation.

Expected multipart/form-data fields for /predict:
    - image: RGB image (required)
    - depth: Depth map (required, meters or millimeters)
    - mask: Optional binary mask
    - fx, fy, cx, cy OR intrinsics JSON (optional, defaults to dataset values)

Example:
    curl -X POST http://localhost:5000/predict \
        -F image=@rgb.png \
        -F depth=@depth.png \
        -F mask=@mask.png \
        -F fx=572.4114 -F fy=573.5704 -F cx=325.2611 -F cy=242.0489
"""

import argparse
from typing import Optional

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from config import cfg

from inference import PoseEstimator


app = Flask(__name__)
estimator: Optional[PoseEstimator] = None


def _load_image(file_storage) -> np.ndarray:
    file_storage.stream.seek(0)
    image = Image.open(file_storage.stream).convert("RGB")
    return np.array(image).astype(np.float32) / 255.0


def _load_depth(file_storage) -> np.ndarray:
    file_storage.stream.seek(0)
    depth = Image.open(file_storage.stream)
    depth_np = np.array(depth).astype(np.float32)
    if depth_np.ndim == 3:
        depth_np = depth_np[..., 0]
    if depth_np.max() > 10.0:
        depth_np /= 1000.0  # assume millimeters
    return depth_np


def _load_mask(file_storage) -> np.ndarray:
    file_storage.stream.seek(0)
    mask = Image.open(file_storage.stream).convert("L")
    mask_np = (np.array(mask).astype(np.float32) > 0).astype(np.float32)
    return mask_np


def _parse_intrinsics(form) -> Optional[np.ndarray]:
    fx = form.get("fx", type=float)
    fy = form.get("fy", type=float)
    cx = form.get("cx", type=float)
    cy = form.get("cy", type=float)
    if None in (fx, fy, cx, cy):
        return None
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


@app.route("/health", methods=["GET"])
def health():
    status = "ready" if estimator else "initializing"
    device = str(estimator.device) if estimator else "n/a"
    return jsonify({"status": status, "model": "Geo6D-Lite", "device": device})


@app.route("/predict", methods=["POST"])
def predict():
    if estimator is None:
        return jsonify({"error": "Model not initialized"}), 503

    image_file = request.files.get("image")
    depth_file = request.files.get("depth")
    if image_file is None or depth_file is None:
        return jsonify({"error": "Both 'image' and 'depth' fields are required."}), 400

    mask_file = request.files.get("mask")
    try:
        image_np = _load_image(image_file)
        depth_np = _load_depth(depth_file)
        mask_np = _load_mask(mask_file) if mask_file else None
        intrinsics = _parse_intrinsics(request.form)

        rot6d, trans, rot_matrix = estimator.predict(
            image_np,
            depth_np,
            intrinsics=intrinsics,
            mask=mask_np,
            return_matrix=True,
        )

        return jsonify(
            {
                "status": "success",
                "rotation_6d": rot6d.tolist(),
                "translation": trans.tolist(),
                "rotation_matrix": rot_matrix.tolist(),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    return (
        jsonify(
            {
                "error": "Batch prediction endpoint is unavailable. "
                "Use /predict per sample with RGB + depth + intrinsics."
            }
        ),
        501,
    )


@app.route("/info", methods=["GET"])
def info():
    device = str(estimator.device) if estimator else "n/a"
    return jsonify(
        {
            "model": "Geo6D-Lite",
            "architecture": f"{cfg.model.backbone}+Geo6DNet",
            "input_size": [cfg.model.input_res, cfg.model.input_res],
            "requires_depth": cfg.model.use_depth,
            "geo_channels": cfg.model.geo_channels,
            "device": device,
        }
    )


def create_parser():
    parser = argparse.ArgumentParser(description="Geo6D-Lite REST API")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/epoch_39.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host/IP to bind to")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--object_id",
        type=str,
        default="05",
        help="LineMOD object ID for default intrinsics",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print("Initializing Geo6D-Lite API Server...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Object ID: {args.object_id}")

    estimator = PoseEstimator(args.checkpoint, device=args.device, object_id=args.object_id)

    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Endpoints: /health, /info, /predict")

    app.run(host=args.host, port=args.port, debug=args.debug)
