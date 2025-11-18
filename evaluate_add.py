import os, json, torch, numpy as np, argparse
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from dataset import LineMODDriveMini
from utils.metrics import add_score, compute_add_accuracy_metrics
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_with_add(model, dataloader, model_points, add_threshold=0.10):
    """
    Evaluate model using ADD (Average Distance of model points) score.
    
    Args:
        model: Trained model
        dataloader: DataLoader for test set
        model_points: Model points in object coordinates (N, 3) as numpy array
        add_threshold: ADD threshold in meters for accuracy calculation (default: 0.10 = 10cm)
    """
    model.eval()
    add_scores = []
    print_debug = True  # Print first 5 samples for debugging
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            imgs, depths, Ks = (
                batch["img"].to(device),
                batch["depth"].to(device),
                batch["K"].to(device),
            )
            R_gt, t_gt = batch["R"].cpu().numpy(), batch["t"].cpu().numpy()

            # Pass GT mask to model for proper confidence-weighted pooling
            mask_for_model = batch.get("mask", None)
            if mask_for_model is not None:
                mask_for_model = mask_for_model.to(device)
            preds = model(imgs, depths, Ks, mask=mask_for_model)
            R_pred = preds["R"].cpu().numpy()
            t_pred = preds["t_off"].cpu().numpy()

            for i in range(len(R_gt)):
                # Compute ADD score for this sample
                add = add_score(R_gt[i], t_gt[i], R_pred[i], t_pred[i], model_points)
                add_scores.append(add)

                # Debug: Print first 5 samples
                if print_debug and sample_count < 5:
                    print(f"\n🔍 Sample {sample_count} Debug:")
                    print(f"   GT Translation (m): {t_gt[i]}")
                    print(f"   Pred Translation (m): {t_pred[i]}")
                    print(f"   ADD Score: {add:.4f} m ({add * 100:.2f} cm)")
                    sample_count += 1
                    if sample_count >= 5:
                        print_debug = False

    # Compute accuracy metrics
    accuracy_metrics = compute_add_accuracy_metrics(add_scores, add_threshold=add_threshold)

    print(f"\n📊 ADD Score Statistics:")
    print(f"   Mean ADD: {accuracy_metrics['mean_add']:.4f} m ({accuracy_metrics['mean_add'] * 100:.2f} cm)")
    print(f"   Median ADD: {accuracy_metrics['median_add']:.4f} m ({accuracy_metrics['median_add'] * 100:.2f} cm)")
    print(f"   Std ADD: {accuracy_metrics['std_add']:.4f} m ({accuracy_metrics['std_add'] * 100:.2f} cm)")
    print(
        f"\n✅ ADD Accuracy (Threshold: {add_threshold * 100:.1f} cm):"
    )
    print(
        f"   ADD Accuracy: {accuracy_metrics['add_accuracy']:.2f}% ({accuracy_metrics['add_correct']}/{accuracy_metrics['total_samples']})"
    )

    return accuracy_metrics


def _create_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Geo6D model using ADD score")
    parser.add_argument(
        "--object_ids",
        type=str,
        default="05",
        help="Comma-separated object IDs (e.g., 05,06,07)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: use cfg.DEFAULT_CHECKPOINT)",
    )
    parser.add_argument(
        "--add_thresh",
        type=float,
        default=0.10,
        help="ADD threshold in meters for accuracy (default: 0.10 = 10cm)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit test set size (default: use all samples)",
    )
    parser.add_argument(
        "--num_model_points",
        type=int,
        default=None,
        help="Number of model points to use (default: use cfg.train.num_model_points)",
    )
    return parser


def main(arg_list=None):
    parser = _create_arg_parser()
    args = parser.parse_args(arg_list)

    object_ids = args.object_ids.split(",")
    
    # For now, we support single object evaluation (model points are object-specific)
    if len(object_ids) > 1:
        print("⚠️ Warning: Multiple objects not yet supported. Using first object:", object_ids[0])
    object_id = object_ids[0]
    
    # Load model points for the object
    num_points = args.num_model_points if args.num_model_points else cfg.train.num_model_points
    print(f"📦 Loading model points for object {object_id} (num_points={num_points})...")
    from utils.model_points import load_or_sample_model_points
    model_points_tensor = load_or_sample_model_points(num_points, object_id=object_id)
    model_points = model_points_tensor.numpy()  # Convert to numpy for ADD computation
    print(f"✅ Loaded {model_points.shape[0]} model points")
    
    # Load test dataset
    test_set = LineMODDriveMini(
        object_ids=[object_id], split="test", max_per_obj=args.max_samples
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Initialize model
    backbone = ResNetBackbone(
        pretrained=False,
        out_channels=cfg.model.feat_dim,
        backbone_type=cfg.model.backbone,
    )
    geo_channels = getattr(cfg.model, "geo_channels", 12)
    model = Geo6DNet(backbone, geo_channels=geo_channels).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint if args.checkpoint else cfg.DEFAULT_CHECKPOINT
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"✅ Loaded checkpoint from {ckpt_path}")
    else:
        print("⚠️ No checkpoint found. Using randomly initialized model.")

    print(f"\n🧪 Evaluating on object: {object_id}")
    print(f"📊 Using ADD threshold: {args.add_thresh * 100:.1f} cm\n")
    
    # Evaluate
    accuracy_metrics = evaluate_with_add(
        model,
        test_loader,
        model_points,
        add_threshold=args.add_thresh,
    )
    
    # Save results to JSON
    results_path = os.path.join(cfg.BASE_DIR, "checkpoints", f"add_eval_{os.path.basename(ckpt_path).replace('.pth', '')}.json")
    with open(results_path, "w") as f:
        json.dump(accuracy_metrics, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")


if __name__ == "__main__":
    main()

