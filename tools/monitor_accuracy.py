#!/usr/bin/env python3
"""
Monitor training progress and run evaluation for new epochs.
Saves per-epoch evaluation results to `checkpoints/eval_epoch_{epoch}.json`.

Usage:
  python3 tools/monitor_accuracy.py --once --object_ids 05
  python3 tools/monitor_accuracy.py          # runs continuously (polling)
"""

import json
import time
import argparse
import subprocess
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")
TRAIN_LOG = CHECKPOINT_DIR / "train_log.json"
EVAL_RESULT_NAME = "eval_results.json"


def read_logged_epochs():
    if not TRAIN_LOG.exists():
        return []
    with open(TRAIN_LOG, "r") as f:
        data = json.load(f)
    return [int(item["epoch"]) for item in data if "epoch" in item]


def eval_epoch(epoch, object_ids):
    """Run evaluate.py for the epoch and save result file to checkpoints/eval_epoch_{epoch}.json
    Note: train_log stores 1-based epoch numbers; checkpoint files are named epoch_{epoch-1}.pth
    """
    ckpt_idx = epoch - 1
    ckpt_path = CHECKPOINT_DIR / f"epoch_{ckpt_idx}.pth"

    if not ckpt_path.exists():
        print(f"Checkpoint not found for epoch {epoch} -> expected {ckpt_path}")
        return False

    # Remove any previous generic eval_results to avoid confusion
    generic = CHECKPOINT_DIR / EVAL_RESULT_NAME
    if generic.exists():
        try:
            generic.unlink()
        except Exception:
            pass

    cmd = (
        ["python3", "evaluate.py", "--object_ids"]
        + object_ids
        + ["--checkpoint", str(ckpt_path), "--save_dir", str(CHECKPOINT_DIR)]
    )
    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("evaluate.py failed:", e)
        return False

    # Move generic eval_results.json to per-epoch file
    if generic.exists():
        target = CHECKPOINT_DIR / f"eval_epoch_{epoch}.json"
        # Avoid overwriting
        if target.exists():
            print(f"Target {target} already exists, skipping move")
        else:
            generic.rename(target)
            print(f"Saved evaluation results to {target}")
        return True
    else:
        print("evaluate.py did not produce expected eval_results.json")
        return False


def main(poll_interval, object_ids, once):
    print(f"Monitoring {TRAIN_LOG} every {poll_interval}s for new epochs")
    evaluated = set()

    # If run once, only evaluate the latest logged epoch (user expectation)
    if once:
        epochs = read_logged_epochs()
        if not epochs:
            print("No logged epochs found.")
            return
        latest = max(epochs)
        target = CHECKPOINT_DIR / f"eval_epoch_{latest}.json"
        if target.exists():
            print(f"Latest epoch {latest} already evaluated ({target}).")
            return
        print(f"Found latest logged epoch {latest}; evaluating...")
        ok = eval_epoch(latest, object_ids)
        if not ok:
            print(f"Evaluation failed for epoch {latest}.")
        return

    while True:
        epochs = read_logged_epochs()
        for e in sorted(epochs):
            if e in evaluated:
                continue
            target = CHECKPOINT_DIR / f"eval_epoch_{e}.json"
            if target.exists():
                evaluated.add(e)
                continue
            print(f"Found logged epoch {e}; evaluating...")
            ok = eval_epoch(e, object_ids)
            if ok:
                evaluated.add(e)
            else:
                print(f"Evaluation failed for epoch {e}, will retry later")
        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--object_ids", nargs="+", default=["05"], help="Object ids")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()
    main(args.poll, args.object_ids, args.once)
