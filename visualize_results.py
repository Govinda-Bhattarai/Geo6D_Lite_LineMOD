#!/usr/bin/env python3
"""
Visualization script for Geo6D training and evaluation results.
Plots training loss, accuracy trends, and generates evaluation report.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load training log
TRAIN_LOG = Path("checkpoints/train_log.json")
with open(TRAIN_LOG, "r") as f:
    train_data = json.load(f)

epochs = [item["epoch"] for item in train_data]
losses = [item["avg_loss"] for item in train_data]

# Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Loss
axes[0].plot(epochs, losses, "b-", linewidth=2, marker="o", markersize=4)
axes[0].axvline(
    x=40, color="g", linestyle="--", linewidth=2, label="Best Model (Epoch 40)"
)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Average Loss", fontsize=12)
axes[0].set_title("Training Loss Over Time", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Accuracy Metrics (manual from evaluation results)
eval_epochs = [14, 30, 36, 37, 40, 41]
overall_accs = [17.81, 24.16, 68.98, 72.91, 78.43, 64.55]
rotation_accs = [27.34, 26.59, 74.67, 80.69, 81.77, 74.33]
translation_accs = [65.05, 90.22, 92.73, 90.89, 95.90, 86.79]

axes[1].plot(
    eval_epochs, overall_accs, "o-", linewidth=2, label="Overall", markersize=8
)
axes[1].plot(
    eval_epochs, rotation_accs, "s-", linewidth=2, label="Rotation", markersize=8
)
axes[1].plot(
    eval_epochs, translation_accs, "^-", linewidth=2, label="Translation", markersize=8
)
axes[1].axvline(x=40, color="g", linestyle="--", linewidth=2, alpha=0.5)
axes[1].axhline(
    y=90, color="r", linestyle=":", linewidth=2, alpha=0.5, label="90% Target"
)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Accuracy (%)", fontsize=12)
axes[1].set_title("Accuracy Metrics by Epoch", fontsize=14, fontweight="bold")
axes[1].set_ylim([0, 105])
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("checkpoints/training_summary.png", dpi=150, bbox_inches="tight")
print("✅ Saved visualization to checkpoints/training_summary.png")

# Print summary report
print("\n" + "=" * 70)
print("GEO6D LITE LINEMOD - FINAL TRAINING SUMMARY")
print("=" * 70)
print(f"\nTotal Epochs Trained: {epochs[-1]}")
print(f"Training Loss Progress: {losses[0]:.4f} → {losses[-1]:.4f}")
print("\nBest Model: Epoch 40")
print("  ✅ Overall Accuracy: 78.43%")
print("  ✅ Rotation Accuracy: 81.77%")
print("  ✅ Translation Accuracy: 95.90%")
print("  ✅ Mean Rotation Error: 6.88°")
print("  ✅ Mean Translation Error: 4.54 cm")
print(f"\nCheckpoint: checkpoints/epoch_39.pth")
print(f"Backup: checkpoints/epoch_36_backup_best.pth")
print("=" * 70)
