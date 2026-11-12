"""
Cross-Sensor Statistical Analysis Runner
Compares Sentinel-2 MSI and Landsat 8/9 OLI glacier segmentation outputs.
Calculates paired t-test, Wilcoxon signed-rank test, 95% Bootstrap CIs, Cohen's d, and Bland-Altman agreement.
"""

import os
import sys
import argparse

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.statistical_sensor_analysis import main as run_stat_analysis


def main():
    parser = argparse.ArgumentParser(description="Cross-Sensor Statistical Analysis")
    parser.add_argument("--s2-dir", type=str, default="data/satellite_patches/s2_patches")
    parser.add_argument("--landsat-dir", type=str, default="data/satellite_patches/landsat_patches")
    parser.add_argument("--mask-dir", type=str, default="data/satellite_patches/masks")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/statistical_analysis")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    args = parser.parse_args()

    sys.argv = [
        "statistical_sensor_analysis.py",
        "--s2-dir", args.s2_dir,
        "--landsat-dir", args.landsat_dir,
        "--mask-dir", args.mask_dir,
        "--checkpoint", args.checkpoint,
        "--output-dir", args.output_dir,
        "--threshold", str(args.threshold),
        "--bootstrap-iters", str(args.bootstrap_iters)
    ]
    run_stat_analysis()


if __name__ == "__main__":
    main()
