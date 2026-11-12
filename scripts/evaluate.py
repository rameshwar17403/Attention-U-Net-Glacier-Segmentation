import os
import sys
import argparse
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from models.attention_unet import AttentionUNet
from src.dataset import GlacierDataset
from src.losses import HybridBCEDiceLoss
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Genuine Evaluation of Attention U-Net")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Path to model weights")
    parser.add_argument("--data-dir", type=str, default="data/patches", help="Path to dataset directory")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification decision threshold")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate: 'test', 'val', 'train', or 'all'")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Directory for evaluation exports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Evaluation running on device: {device}")

    inv_csv = os.path.join(repo_root, "data", "dataset_inventory.csv")
    split_arg = None if args.split == "all" else args.split
    if os.path.exists(inv_csv) and split_arg:
        print(f"[+] Loading '{split_arg}' split from {inv_csv}")
        dataset = GlacierDataset(args.data_dir, split=split_arg, inventory_csv=inv_csv)
    else:
        dataset = GlacierDataset(args.data_dir)
    if len(dataset) == 0:
        print(f"[!] No patches found in '{args.data_dir}'. Aborting evaluation.")
        return

    print(f"[+] Evaluating on {len(dataset)} genuine patches from '{args.data_dir}'.")

    model = AttentionUNet(in_channels=8, out_channels=1).to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"[!] Error: Model checkpoint '{args.checkpoint}' not found. "
            f"Evaluation using uninitialized or random weights is strictly prohibited. Run scripts/train.py first."
        )
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[+] Successfully loaded trained model weights from: {args.checkpoint}")
    except Exception as e:
        raise RuntimeError(f"[!] Critical Error: Failed loading checkpoint '{args.checkpoint}': {e}")

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    patch_records = []
    preview_sample = None

    with torch.no_grad():
        for idx in range(len(dataset)):
            patch_tensor, mask_tensor = dataset[idx]
            patch_id = os.path.basename(dataset.patch_files[idx]).replace("_patch.npy", "")

            # Shape: (1, 8, H, W)
            input_batch = patch_tensor.unsqueeze(0).to(device)
            logits = model(input_batch)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (probs >= args.threshold).astype(np.uint8)
            gt_bin = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

            tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
            fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
            fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
            tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            eps = 1e-8
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
            iou = tp / (tp + fp + fn + eps)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = (2.0 * precision * recall) / (precision + recall + eps)

            patch_records.append({
                "patch_id": patch_id,
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "ground_truth_glacier_pixels": int(np.sum(gt_bin)),
                "predicted_glacier_pixels": int(np.sum(pred_bin)),
            })

            # Select sample with glacier for visual overlay
            if preview_sample is None and np.sum(gt_bin) > 500:
                raw_patch = patch_tensor.cpu().numpy()
                preview_sample = {
                    "patch_id": patch_id,
                    "patch": raw_patch,
                    "probs": probs,
                    "pred_bin": pred_bin,
                    "gt_bin": gt_bin
                }

    # Cumulative overall metrics
    eps = 1e-8
    ov_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
    ov_iou = total_tp / (total_tp + total_fp + total_fn + eps)
    ov_precision = total_tp / (total_tp + total_fp + eps)
    ov_recall = total_tp / (total_tp + total_fn + eps)
    ov_f1 = (2.0 * ov_precision * ov_recall) / (ov_precision + ov_recall + eps)

    # Save per-patch metrics CSV
    per_patch_df = pd.DataFrame(patch_records)
    per_patch_csv = os.path.join(args.output_dir, "per_patch_evaluation.csv")
    per_patch_df.to_csv(per_patch_csv, index=False)

    # Save overall summary CSV
    summary_df = pd.DataFrame([{
        "dataset_patches": len(dataset),
        "threshold": args.threshold,
        "overall_dice": ov_dice,
        "overall_iou": ov_iou,
        "overall_precision": ov_precision,
        "overall_recall": ov_recall,
        "overall_f1": ov_f1,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "total_true_negatives": total_tn,
    }])
    summary_csv = os.path.join(args.output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Save confusion matrix CSV
    cm_df = pd.DataFrame([
        {"Class": "Actual Glacier", "Pred Glacier (TP)": total_tp, "Pred Background (FN)": total_fn},
        {"Class": "Actual Background", "Pred Glacier (FP)": total_fp, "Pred Background (TN)": total_tn}
    ])
    cm_csv = os.path.join(args.output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv, index=False)

    # Generate visual segmentation overlay preview
    if preview_sample is not None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        p = preview_sample["patch"]
        # False color RGB (NIR: B8, Red: B4, Green: B3)
        rgb = np.stack([p[3], p[2], p[1]], axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        axes[0].imshow(rgb)
        axes[0].set_title(f"(a) False-Color NIR-R-G ({preview_sample['patch_id']})", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(preview_sample["gt_bin"], cmap="Blues")
        axes[1].set_title("(b) RGI 6.0 Reference Mask", fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(preview_sample["probs"], cmap="magma")
        axes[2].set_title(f"(c) Attention U-Net Probability", fontweight="bold")
        axes[2].axis("off")

        # Discrepancy map
        h, w = preview_sample["gt_bin"].shape
        err_map = np.zeros((h, w, 3), dtype=np.float32)
        both_tp = (preview_sample["pred_bin"] == 1) & (preview_sample["gt_bin"] == 1)
        fp_err = (preview_sample["pred_bin"] == 1) & (preview_sample["gt_bin"] == 0)
        fn_err = (preview_sample["pred_bin"] == 0) & (preview_sample["gt_bin"] == 1)

        err_map[both_tp] = [0.1, 0.8, 0.2]   # Green: True Positive
        err_map[fp_err] = [0.9, 0.2, 0.1]    # Red: False Positive (Over-segmentation)
        err_map[fn_err] = [0.2, 0.4, 0.9]    # Blue: False Negative (Omission)

        axes[3].imshow(err_map)
        axes[3].set_title("(d) Spatial Discrepancy Map", fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        overlay_path = os.path.join(args.output_dir, "segmentation_overlay_preview.png")
        plt.savefig(overlay_path, dpi=300)
        plt.close()
        print(f"[+] Saved visual segmentation overlay preview to: {overlay_path}")

    print("\n" + "=" * 60)
    print("        GENUINE ATTENTION U-NET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Patches Evaluated:  {len(dataset)}")
    print(f"Decision Threshold: {args.threshold:.2f}")
    print(f"Overall Dice:       {ov_dice:.4f} (Mean per-patch: {per_patch_df['dice'].mean():.4f})")
    print(f"Overall IoU:        {ov_iou:.4f} (Mean per-patch: {per_patch_df['iou'].mean():.4f})")
    print(f"Overall Precision:  {ov_precision:.4f}")
    print(f"Overall Recall:     {ov_recall:.4f}")
    print(f"Overall F1 Score:   {ov_f1:.4f}")
    print("-" * 60)
    print(f"Confusion Matrix:")
    print(f"  TP: {total_tp:,} | FP: {total_fp:,}")
    print(f"  FN: {total_fn:,} | TN: {total_tn:,}")
    print("=" * 60)
    print(f"[+] Exported results to: {args.output_dir}")


if __name__ == "__main__":
    main()
