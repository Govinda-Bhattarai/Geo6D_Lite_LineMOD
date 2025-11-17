#!/usr/bin/env python3
"""Prepare model_points.npy for all objects found under datasets/Linemod_preprocessed/data

Usage:
    python tools/prepare_all_model_points.py --num_points 500 [--force]

This script loops over object folders and calls tools/prepare_model_points.py for each.
It skips objects where model_points.npy already exists unless --force is provided.
"""
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, "datasets", "Linemod_preprocessed", "data")
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        return 2

    objs = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, d))]
    if len(objs) == 0:
        print("No object folders found under data root.")
        return 1

    helper = os.path.join(os.path.dirname(__file__), "prepare_model_points.py")
    for obj in objs:
        print(f"\n=== Preparing object {obj} ===")
        cmd = [sys.executable, helper, '--object_id', obj, '--num_points', str(args.num_points)]
        if args.force:
            cmd.append('--force')
        print('Running:', ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Failed for {obj}: {e}")

    print("\nAll done.")
    return 0

if __name__ == '__main__':
    import sys
    raise SystemExit(main())
