"""Model-point loader / sampler for LineMOD objects.

Now supports per-object loading via object_id argument and returns a torch.FloatTensor (CPU).
"""

import os
import numpy as np
import torch
from config import cfg


def load_or_sample_model_points(num_points=500, object_id=None):
    """Load or sample model points for a given object_id or the first object found.

    Args:
        num_points (int): number of points to return.
        object_id (str|None): object id folder under datasets/Linemod_preprocessed/data/.

    Returns:
        torch.FloatTensor: (P,3) model points in object coordinates (CPU tensor).
    """
    data_root = os.path.join(cfg.BASE_DIR, "datasets", "Linemod_preprocessed", "data")
    if not os.path.isdir(data_root):
        print("[model_points] LineMOD data root not found, returning random points")
        return torch.from_numpy(
            (np.random.rand(num_points, 3).astype(np.float32) - 0.5)
        )

    if object_id is None:
        objs = [
            d
            for d in sorted(os.listdir(data_root))
            if os.path.isdir(os.path.join(data_root, d))
        ]
        if len(objs) == 0:
            print("[model_points] No object folders found, returning random points")
            return torch.from_numpy(
                (np.random.rand(num_points, 3).astype(np.float32) - 0.5)
            )
        obj = objs[0]
    else:
        obj = str(object_id)

    root = os.path.join(data_root, obj)
    npy_path = os.path.join(root, "model_points.npy")

    if os.path.exists(npy_path):
        try:
            pts = np.load(npy_path)
            pts = pts.astype(np.float32)
            if pts.shape[0] >= num_points:
                idx = np.random.choice(pts.shape[0], num_points, replace=False)
                pts = pts[idx]
            else:
                idx = np.random.choice(pts.shape[0], num_points, replace=True)
                pts = pts[idx]
            print(f"[model_points] Loaded {pts.shape[0]} points from {npy_path}")
            return torch.from_numpy(pts)
        except Exception as e:
            print(f"[model_points] Failed to load {npy_path}: {e}")

    # Try to load a mesh and sample points (requires trimesh)
    for mesh_name in ("model.ply", "model.obj", "mesh.ply", "model.pcd"):
        mesh_path = os.path.join(root, mesh_name)
        if os.path.exists(mesh_path):
            try:
                import trimesh  # type: ignore

                mesh = trimesh.load(mesh_path, force="mesh")
                if getattr(mesh, "is_empty", False):
                    continue
                pts = mesh.sample(num_points)
                pts = np.asarray(pts, dtype=np.float32)
                try:
                    np.save(npy_path, pts)
                    print(
                        f"[model_points] Sampled {pts.shape[0]} points from mesh and saved to {npy_path}"
                    )
                except Exception:
                    pass
                return torch.from_numpy(pts)
            except Exception as e:
                print(f"[model_points] trimesh failed to load/sample {mesh_path}: {e}")
                break

    # Final fallback: random sampling in unit cube (not a correct model)
    print(
        f"[model_points] No model file found for object {obj}; returning random placeholder points (not saved to disk). Place a mesh (model.ply/model.obj) in the object folder and re-run tools/prepare_model_points.py to create a proper model_points.npy"
    )
    pts = np.random.rand(num_points, 3).astype(np.float32) - 0.5
    # Do NOT save placeholder points to disk to avoid accidentally overwriting real model_points.npy
    return torch.from_numpy(pts)
