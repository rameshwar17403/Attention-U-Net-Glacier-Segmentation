import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from models.attention_unet import AttentionUNet
from src.dataset import GlacierDataset
from src.losses import HybridBCEDiceLoss, FocalTverskyLoss
from src.engine import train_epoch, validate_model
from src.utils import seed_everything, load_config


def plot_training_curves(history: pd.DataFrame, out_path: str):
    """Plots training and validation loss, Dice, and IoU curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="#d62728", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("Training & Validation Loss Convergence", fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Metric curve
    axes[1].plot(history["epoch"], history["val_dice"], label="Val Dice Score", color="#2ca02c", linewidth=2)
    axes[1].plot(history["epoch"], history["val_iou"], label="Val IoU Score", color="#9467bd", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Score", fontweight="bold")
    axes[1].set_title("Validation Segmentation Metric Progression", fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved training curves to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Attention U-Net for Glacier Segmentation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--loss", type=str, choices=["hybrid", "focal_tversky"], default="hybrid",
                        help="Loss function: 'hybrid' (BCE+Dice) or 'focal_tversky'")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(42)

    # Determine optimal hardware device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Training on device: {device}")

    data_dir = config["data"]["data_dir"]
    if not os.path.exists(data_dir) or len([f for f in os.listdir(data_dir) if f.endswith("_patch.npy")]) == 0:
        print(f"[!] No patch data found in '{data_dir}'. Please run scripts/acquire_satellite_data.py first.")
        return

    dataset = GlacierDataset(data_dir)
    print(f"[+] Loaded GlacierDataset with {len(dataset)} genuine 8-channel patches.")

    val_size = max(1, int(len(dataset) * (1.0 - config["data"]["train_val_split"])))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    batch_size = args.batch_size or config["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"[+] Train split: {train_size} patches | Validation split: {val_size} patches")

    model = AttentionUNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"]["features"]
    ).to(device)

    # Loss selection
    if args.loss == "focal_tversky":
        criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33).to(device)
        print("[+] Using Focal Tversky Loss (alpha=0.3, beta=0.7, gamma=1.33)")
    else:
        criterion = HybridBCEDiceLoss(alpha=config["training"]["alpha_bce_dice"]).to(device)
        print(f"[+] Using Hybrid BCE-Dice Loss (alpha={config['training']['alpha_bce_dice']})")

    lr = args.lr or config["training"]["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    epochs = args.epochs or config["training"]["epochs"]

    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_dir = os.path.join("outputs", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    history_records = []
    best_dice = 0.0

    print(f"\n[*] Starting training loop for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t_loss, t_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = validate_model(model, val_loader, criterion, device, threshold=0.50)

        record = {
            "epoch": epoch,
            "train_loss": t_loss,
            "train_dice": t_dice,
            "val_loss": v_loss,
            "val_dice": v_metrics["dice"],
            "val_iou": v_metrics["iou"],
            "val_precision": v_metrics["precision"],
            "val_recall": v_metrics["recall"],
        }
        history_records.append(record)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {t_loss:.4f} (Dice: {t_dice:.4f}) | "
              f"Val Loss: {v_loss:.4f} (Dice: {v_metrics['dice']:.4f}, IoU: {v_metrics['iou']:.4f}, "
              f"Prec: {v_metrics['precision']:.4f}, Rec: {v_metrics['recall']:.4f})")

        if v_metrics["dice"] > best_dice:
            best_dice = v_metrics["dice"]
            ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved new best model checkpoint with Dice: {best_dice:.4f}")

    # Export history and curves
    history_df = pd.DataFrame(history_records)
    csv_path = os.path.join(eval_dir, "training_history.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"\n[+] Exported training metrics log to: {csv_path}")

    curve_path = os.path.join(eval_dir, "training_curves.png")
    plot_training_curves(history_df, curve_path)

    print(f"[+] Training complete! Best Validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
