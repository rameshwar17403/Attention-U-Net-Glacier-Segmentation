"""
Attention U-Net Quantitative Evaluation Script
Evaluates trained model weights against independent ground truth test split.
Exports confusion matrix, per-patch metrics, and cumulative Dice/IoU/Precision/Recall/F1.
"""

import os
import sys
import argparse

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.evaluate import main as run_eval


def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Launcher")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--data-dir", type=str, default="data/patches")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    sys.argv = [
        "evaluate.py",
        "--checkpoint", args.checkpoint,
        "--data-dir", args.data_dir,
        "--threshold", str(args.threshold),
        "--split", args.split,
        "--output-dir", args.output_dir
    ]
    run_eval()


if __name__ == "__main__":
    main()
