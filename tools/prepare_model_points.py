#!/usr/bin/env python3
"""Prepare model_points.npy for a LineMOD object folder.

Usage:
    python tools/prepare_model_points.py --object_id 05 --num_points 500 [--force]

Behavior:
- If model_points.npy exists and --force not used: print stats and exit.
- If a mesh file (model.ply/model.obj/mesh.ply) exists, sample surface points using trimesh and save model_points.npy.
- If sampled points appear to be in large units (mean extent > 2.0), assume units are mm and convert to meters (/1000) before saving.
- If no mesh found and --fallback, create random placeholder points (NOT recommended) and save.
"""

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_id", type=str, default="05")
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing model_points.npy"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Create random placeholder points if no mesh found",
    )
    args = parser.parse_args()

    # locate dataset root relative to this repo
    # repo root should be the Geo6D_Lite_LineMOD folder (one level up from tools/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, "datasets", "Linemod_preprocessed", "data")
    obj = args.object_id
    obj_dir = os.path.join(data_root, obj)
    if not os.path.isdir(obj_dir):
        print(f"[ERROR] Object folder not found: {obj_dir}")
        return 2

    npy_path = os.path.join(obj_dir, "model_points.npy")
    if os.path.exists(npy_path) and not args.force:
        pts = np.load(npy_path)
        print(
            f"model_points.npy already exists ({npy_path}) — use --force to overwrite"
        )
        print(
            f"shape={pts.shape} dtype={pts.dtype} min={pts.min():.6f} max={pts.max():.6f} mean_extent={(pts.max(0) - pts.min(0)).mean():.6f}"
        )
        return 0

    # try to find mesh
    mesh_names = ["model.ply", "model.obj", "mesh.ply", "model.pcd"]
    mesh_path = None
    for mn in mesh_names:
        p = os.path.join(obj_dir, mn)
        if os.path.exists(p):
            mesh_path = p
            break

    if mesh_path is None:
        if args.fallback:
            print("No mesh found, creating random placeholder points (fallback)")
            pts = np.random.rand(args.num_points, 3).astype(np.float32) - 0.5
            np.save(npy_path, pts)
            print(f"Saved placeholder model_points.npy -> {npy_path}")
            return 0
        else:
            print("No mesh found and --fallback not set. Aborting.")
            return 1

    # sample from mesh using trimesh
    try:
        import trimesh  # type: ignore
    except ImportError:
        print(
            "[ERROR] trimesh not installed. Install with: pip install trimesh rtree pyglet"
        )
        return 3

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    if getattr(mesh, "is_empty", False):
        print("[ERROR] Mesh appears empty")
        return 4

    print(f"Sampling {args.num_points} points from mesh surface...")
    pts = mesh.sample(args.num_points)
    pts = np.asarray(pts, dtype=np.float32)

    # check bounding extents to guess units
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extent = maxs - mins
    mean_extent = float(extent.mean())
    print(f"Sampled points shape={pts.shape} mean_extent={mean_extent:.6f}")

    # If extent seems large (likely mm), convert to meters
    if mean_extent > 2.0:
        print(
            "Mean extent > 2.0 — assuming units are millimeters. Converting to meters (/1000)"
        )
        pts = pts.astype(np.float32) / 1000.0
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        extent = maxs - mins
        print(f"Converted extent mean={float(extent.mean()):.6f}")

    # save
    try:
        np.save(npy_path, pts)
        print(f"Saved model_points.npy -> {npy_path}")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to save {npy_path}: {e}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
