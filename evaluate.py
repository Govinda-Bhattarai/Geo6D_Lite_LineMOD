import os, json, torch, numpy as np, argparse
from models.backbone import ResNetBackbone
from models.pose_head import Geo6DNet
from dataset import LineMODDriveMini
from utils.metrics import rotation_error, translation_error, compute_accuracy_metrics
from config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, rot_threshold=10.0, trans_threshold=0.10):
    model.eval()
    rot_errs, trans_errs = [], []
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

            # Pass GT mask to model for proper confidence-weighted pooling (CRITICAL FIX)
            mask_for_model = batch.get("mask", None)
            if mask_for_model is not None:
                mask_for_model = mask_for_model.to(device)
            preds = model(imgs, depths, Ks, mask=mask_for_model)
            R_pred = preds["R"].cpu().numpy()
            t_pred = preds["t_off"].cpu().numpy()

            for i in range(len(R_gt)):
                # CRITICAL DEBUG: Print first 5 samples to check scale
                if print_debug and sample_count < 5:
                    print(f"\n🔍 Sample {sample_count} Debug:")
                    print(f"   GT Translation (m): {t_gt[i]}")
                    print(f"   Pred Translation (m): {t_pred[i]}")
                    print(
                        f"   GT Translation magnitude: {np.linalg.norm(t_gt[i]):.4f} m"
                    )
                    print(
                        f"   Pred Translation magnitude: {np.linalg.norm(t_pred[i]):.4f} m"
                    )
                    print(
                        f"   Translation difference: {np.linalg.norm(t_gt[i] - t_pred[i]):.4f} m"
                    )
                    print(
                        f"   Translation error (cm): {translation_error(t_gt[i], t_pred[i]):.2f} cm"
                    )
                    print(
                        f"   Rotation error: {rotation_error(R_gt[i], R_pred[i]):.2f}°"
                    )
                    sample_count += 1
                    if sample_count >= 5:
                        print_debug = False

                rot_errs.append(rotation_error(R_gt[i], R_pred[i]))
                trans_errs.append(translation_error(t_gt[i], t_pred[i]))

    mean_rot = np.mean(rot_errs)
    mean_trans = np.mean(trans_errs)

    # Compute accuracy metrics with configurable thresholds
    accuracy_metrics = compute_accuracy_metrics(
        rot_errs,
        trans_errs,
        rot_threshold=rot_threshold,
        trans_threshold=trans_threshold * 100.0,
    )

    print(f"📊 Mean Rotation Error: {mean_rot:.2f}°")
    print(f"📊 Mean Translation Error: {mean_trans:.2f} cm")
    print(
        f"\n✅ Accuracy Metrics (Threshold: {rot_threshold}° rotation, {trans_threshold * 100:.0f} cm translation):"
    )
    print(
        f"   Rotation Accuracy: {accuracy_metrics['rotation_accuracy']:.2f}% ({accuracy_metrics['rotation_correct']}/{accuracy_metrics['total_samples']})"
    )
    print(
        f"   Translation Accuracy: {accuracy_metrics['translation_accuracy']:.2f}% ({accuracy_metrics['translation_correct']}/{accuracy_metrics['total_samples']})"
    )
    print(
        f"   Overall Accuracy: {accuracy_metrics['overall_accuracy']:.2f}% ({accuracy_metrics['overall_correct']}/{accuracy_metrics['total_samples']})"
    )

    return mean_rot, mean_trans, accuracy_metrics


def _create_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Geo6D model on test set")
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
        "--rot_thresh",
        type=float,
        default=10.0,
        help="Rotation threshold in degrees (default: 10)",
    )
    parser.add_argument(
        "--trans_thresh",
        type=float,
        default=0.10,
        help="Translation threshold in meters (default: 0.10 = 10cm)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit test set size (default: use all samples)",
    )
    return parser


def main(arg_list=None):
    parser = _create_arg_parser()
    args = parser.parse_args(arg_list)

    object_ids = args.object_ids.split(",")
    test_set = LineMODDriveMini(
        object_ids=object_ids, split="test", max_per_obj=args.max_samples
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    backbone = ResNetBackbone(
        pretrained=False,
        out_channels=cfg.model.feat_dim,
        backbone_type=cfg.model.backbone,  # Use configured backbone type
    )
    geo_channels = getattr(cfg.model, "geo_channels", 12)
    model = Geo6DNet(backbone, geo_channels=geo_channels).to(device)

    ckpt_path = args.checkpoint if args.checkpoint else cfg.DEFAULT_CHECKPOINT
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"✅ Loaded checkpoint from {ckpt_path}")
    else:
        print("⚠️ No checkpoint found. Using randomly initialized model.")

    print(f"\nEvaluating on object(s): {object_ids}")
    print(
        f"Using thresholds: {args.rot_thresh}° rotation, {args.trans_thresh * 100:.1f}cm translation\n"
    )
    evaluate(
        model,
        test_loader,
        rot_threshold=args.rot_thresh,
        trans_threshold=args.trans_thresh,
    )


if __name__ == "__main__":
    main()
