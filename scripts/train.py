"""
Attention U-Net Training Pipeline for Multispectral Glacier Segmentation
Sentinel-2 8-Channel Reflectance & Indices (NASA HLS 30m UTM 44N Grid)
Strict Temporal Scene-Level Split (Zero Data Leakage)

Author: Rameshwar Chandravanshi (Pair Programming with Antigravity)
Project: Attention U-Net Multispectral Glacier Segmentation Framework
"""

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
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from models.attention_unet import AttentionUNet
from src.dataset import GlacierDataset
from src.losses import HybridBCEDiceLoss, FocalTverskyLoss
from src.engine import train_epoch, validate_model
from src.utils import seed_everything, load_config


def plot_training_curves(history: pd.DataFrame, out_path: str):
    """Plots training and validation loss, Dice, IoU, Precision, and Recall curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss curve
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2.2)
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="#d62728", linewidth=2.2, linestyle="--")
    axes[0].set_xlabel("Epoch", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Loss", fontweight="bold", fontsize=11)
    axes[0].set_title("Training & Validation Loss Convergence", fontweight="bold", fontsize=12)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10)

    # 2. Dice & IoU curve
    axes[1].plot(history["epoch"], history["train_dice"], label="Train Dice", color="#17becf", linewidth=1.8, linestyle=":")
    axes[1].plot(history["epoch"], history["val_dice"], label="Val Dice Score", color="#2ca02c", linewidth=2.2)
    axes[1].plot(history["epoch"], history["val_iou"], label="Val IoU Score", color="#9467bd", linewidth=2.2, linestyle="--")
    axes[1].set_xlabel("Epoch", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Score", fontweight="bold", fontsize=11)
    axes[1].set_title("Overlap Metrics (Dice & IoU)", fontweight="bold", fontsize=12)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10)

    # 3. Precision & Recall curve
    axes[2].plot(history["epoch"], history["val_precision"], label="Val Precision", color="#ff7f0e", linewidth=2.2)
    axes[2].plot(history["epoch"], history["val_recall"], label="Val Recall", color="#8c564b", linewidth=2.2, linestyle="--")
    axes[2].set_xlabel("Epoch", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("Score", fontweight="bold", fontsize=11)
    axes[2].set_title("Validation Precision & Recall Progression", fontweight="bold", fontsize=12)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved training convergence curves to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Attention U-Net for Glacier Segmentation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--loss", type=str, choices=["hybrid", "focal_tversky"], default="hybrid",
                        help="Loss function: 'hybrid' (BCE+Dice) or 'focal_tversky'")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    args = parser.parse_args()

    config_path = os.path.join(repo_root, args.config) if not os.path.isabs(args.config) else args.config
    config = load_config(config_path)
    seed_everything(42)

    # Determine optimal hardware device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Training on compute device: {device}")

    data_dir = os.path.join(repo_root, config["data"]["data_dir"])
    inv_csv = os.path.join(repo_root, "data", "dataset_inventory.csv")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"[!] Dataset directory '{data_dir}' not found.")

    # Load datasets using strict scene-level splitting if inventory exists
    if os.path.exists(inv_csv):
        print(f"[+] Loading scene-level split from inventory: {inv_csv}")
        train_ds = GlacierDataset(data_dir, split="train", inventory_csv=inv_csv)
        val_ds = GlacierDataset(data_dir, split="val", inventory_csv=inv_csv)
        print(f"[+] Scene-split loaded: Train = {len(train_ds)} patches | Val = {len(val_ds)} patches")
    else:
        full_ds = GlacierDataset(data_dir)
        val_size = max(1, int(len(full_ds) * (1.0 - config["data"]["train_val_split"])))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        print(f"[!] Warning: Inventory not found. Fallback random split: Train = {train_size} | Val = {val_size}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty dataset split: Train={len(train_ds)}, Val={len(val_ds)}")

    batch_size = args.batch_size or config["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[+] DataLoader configured: Batch size = {batch_size} | Train batches = {len(train_loader)} | Val batches = {len(val_loader)}")

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
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    ckpt_dir = os.path.join(repo_root, config["training"]["checkpoint_dir"])
    model_dir = os.path.join(repo_root, "models", "attention_unet")
    eval_dir = os.path.join(repo_root, "outputs", "evaluation")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    history_records = []
    best_dice = 0.0
    patience = args.patience
    patience_counter = 0

    print(f"\n[*] Commencing Attention U-Net training loop ({epochs} epochs, lr={lr}, device={device})...")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        t_loss, t_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = validate_model(model, val_loader, criterion, device, threshold=0.50)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": t_loss,
            "train_dice": t_dice,
            "val_loss": v_loss,
            "val_dice": v_metrics["dice"],
            "val_iou": v_metrics["iou"],
            "val_precision": v_metrics["precision"],
            "val_recall": v_metrics["recall"],
            "lr": float(scheduler.get_last_lr()[0])
        }
        history_records.append(record)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {t_loss:.4f} (Dice: {t_dice:.4f}) | "
              f"Val Loss: {v_loss:.4f} (Dice: {v_metrics['dice']:.4f}, IoU: {v_metrics['iou']:.4f}, "
              f"Prec: {v_metrics['precision']:.4f}, Rec: {v_metrics['recall']:.4f})")

        if v_metrics["dice"] > best_dice:
            best_dice = v_metrics["dice"]
            patience_counter = 0
            ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            model_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), model_path)
            print(f"  -> ⭐ Saved new best model checkpoint! Val Dice: {best_dice:.4f} (IoU: {v_metrics['iou']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs).")
                break

    # Export history and curves
    history_df = pd.DataFrame(history_records)
    csv_path = os.path.join(eval_dir, "training_history.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"\n[+] Exported training metrics log to: {csv_path}")

    curve_path = os.path.join(eval_dir, "training_curves.png")
    plot_training_curves(history_df, curve_path)

    print(f"[+] Training successfully completed! Best Validation Dice Score: {best_dice:.4f}")
    return history_df, best_dice


if __name__ == "__main__":
    main()
