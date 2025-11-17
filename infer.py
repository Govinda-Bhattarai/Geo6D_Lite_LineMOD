#!/usr/bin/env python3
"""
Inference script for Geo6D-Lite pose estimation model.
Evaluates the best trained model on test data or custom images.

Usage:
    # Evaluate on test dataset
    python3 infer.py --object_ids 05 --checkpoint checkpoints/epoch_39.pth

    # Evaluate on specific object
    python3 infer.py --object_ids 05 --checkpoint checkpoints/epoch_39.pth --mode eval
"""

import argparse
from typing import List

try:
    import torch
except ImportError:
    torch = None

from evaluate import main as evaluate_main


def create_parser():
    parser = argparse.ArgumentParser(
        description="Inference script for Geo6D-Lite pose estimation"
    )
    parser.add_argument(
        "--object_ids",
        type=int,
        nargs="+",
        required=True,
        help="Object ID(s) to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (e.g., checkpoints/epoch_39.pth)",
    )
    parser.add_argument(
        "--rotation_threshold",
        type=float,
        default=10.0,
        help="Rotation error threshold in degrees (default: 10)",
    )
    parser.add_argument(
        "--translation_threshold",
        type=float,
        default=10.0,
        help="Translation error threshold in cm (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available)",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    print("Geo6D-Lite Pose Estimation Inference")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Object IDs: {args.object_ids}")
    print(f"Rotation threshold: {args.rotation_threshold}°")
    print(f"Translation threshold: {args.translation_threshold} cm")
    print(f"Device: {args.device}")
    print("=" * 50)

    # evaluate.py expects translation threshold in meters and comma-separated IDs
    eval_args: List[str] = [
        "--object_ids",
        ",".join(f"{oid:02d}" if isinstance(oid, int) else str(oid) for oid in args.object_ids),
        "--checkpoint",
        args.checkpoint,
        "--rot_thresh",
        str(args.rotation_threshold),
        "--trans_thresh",
        str(args.translation_threshold / 100.0),
    ]

    evaluate_main(eval_args)


if __name__ == "__main__":
    main()
