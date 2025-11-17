#!/usr/bin/env python3
"""
Visualize dataset samples to verify data loading.
Shows RGB, depth, and mask images from the dataset.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import LineMODDriveMini
from config import cfg

def visualize_sample(dataset, idx=0, save_path=None):
    """
    Visualize a single sample from the dataset.
    
    Args:
        dataset: LineMODDriveMini dataset instance
        idx: Index of sample to visualize
        save_path: Optional path to save the visualization
    """
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return
    
    if idx >= len(dataset):
        print(f"⚠️  Index {idx} out of range. Using index 0 instead.")
        idx = 0
    
    # Get sample
    sample = dataset[idx]
    
    # Convert tensors to numpy for visualization
    img = sample['img'].permute(1, 2, 0).numpy()  # (H, W, C)
    img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
    
    # Depth (if available)
    depth = None
    if sample['depth'] is not None:
        depth = sample['depth'].squeeze().numpy()  # (H, W)
    
    # Mask (if available)
    mask = None
    if sample['mask'] is not None:
        mask = sample['mask'].squeeze().numpy()  # (H, W)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB Image
    axes[0].imshow(img)
    axes[0].set_title(f'RGB Image (Sample {idx})', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Depth Image
    if depth is not None:
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        im = axes[1].imshow(depth_normalized, cmap='jet')
        axes[1].set_title('Depth Map', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, 'No Depth Data', 
                    ha='center', va='center', fontsize=14)
        axes[1].set_title('Depth Map (Not Available)', fontsize=12)
        axes[1].axis('off')
    
    # Mask Image
    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Mask', fontsize=12, fontweight='bold')
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'No Mask Data', 
                    ha='center', va='center', fontsize=14)
        axes[2].set_title('Mask (Not Available)', fontsize=12)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Print sample information
    print(f"\n{'='*60}")
    print(f"Sample {idx} Information:")
    print(f"{'='*60}")
    print(f"Image shape: {img.shape}")
    if depth is not None:
        print(f"Depth shape: {depth.shape}")
        print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    if mask is not None:
        print(f"Mask shape: {mask.shape}")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        print(f"Mask pixels (non-zero): {np.count_nonzero(mask)} / {mask.size}")
    
    print(f"\nPose Information:")
    print(f"Rotation matrix R:\n{sample['R'].numpy()}")
    print(f"Translation t: {sample['t'].numpy()}")
    print(f"Camera intrinsics K:\n{sample['K'].numpy()}")
    print(f"{'='*60}\n")
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_multiple_samples(dataset, num_samples=5, save_dir=None):
    """
    Visualize multiple samples from the dataset.
    
    Args:
        dataset: LineMODDriveMini dataset instance
        num_samples: Number of samples to visualize
        save_dir: Optional directory to save visualizations
    """
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return
    
    num_samples = min(num_samples, len(dataset))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📊 Visualizing {num_samples} samples from dataset...\n")
    
    for i in range(num_samples):
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"sample_{i:03d}.png")
        
        visualize_sample(dataset, idx=i, save_path=save_path)
    
    print(f"✅ Visualized {num_samples} samples!")

def main():
    """Main function to run visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize LineMOD dataset samples")
    parser.add_argument("--object_ids", type=str, nargs="+", default=["05"], 
                       help="Object IDs to visualize (e.g., '05')")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                       help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=10,
                       help="Max samples to load from dataset")
    parser.add_argument("--num_visualize", type=int, default=3,
                       help="Number of samples to visualize")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save visualizations (if None, displays interactively)")
    parser.add_argument("--sample_idx", type=int, default=None,
                       help="Visualize specific sample index (overrides num_visualize)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 Dataset Visualization")
    print("="*60)
    print(f"Object IDs: {args.object_ids}")
    print(f"Split: {args.split}")
    print(f"Max samples to load: {args.max_samples}")
    print("="*60)
    
    # Load dataset
    try:
        dataset = LineMODDriveMini(
            object_ids=args.object_ids,
            split=args.split,
            max_per_obj=args.max_samples
        )
        
        if len(dataset) == 0:
            print("❌ No samples loaded from dataset!")
            return
        
        print(f"✅ Loaded {len(dataset)} samples\n")
        
        # Visualize
        if args.sample_idx is not None:
            # Visualize specific sample
            save_path = None
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(args.save_dir, f"sample_{args.sample_idx:03d}.png")
            visualize_sample(dataset, idx=args.sample_idx, save_path=save_path)
        else:
            # Visualize multiple samples
            visualize_multiple_samples(
                dataset, 
                num_samples=args.num_visualize,
                save_dir=args.save_dir
            )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

