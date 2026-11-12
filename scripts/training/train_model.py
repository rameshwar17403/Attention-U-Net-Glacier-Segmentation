"""
Attention U-Net Training Launcher
Trains 8-channel Attention U-Net on Sentinel-2 cryospheric patches.
Supports Apple Silicon Metal (MPS), NVIDIA CUDA, and CPU.
"""

import os
import sys
import argparse

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.train import main as run_train


def main():
    parser = argparse.ArgumentParser(description="Attention U-Net Training Launcher")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--loss", type=str, default="hybrid", choices=["hybrid", "focal_tversky"])
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Pass through to training main
    sys.argv = [
        "train.py",
        "--config", args.config,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--loss", args.loss,
        "--lr", str(args.lr)
    ]
    run_train()


if __name__ == "__main__":
    main()
