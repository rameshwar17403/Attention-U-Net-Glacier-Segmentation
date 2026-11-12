"""
Visualization and Mapping Script
Generates spatial discrepancy maps, false-color RGB overlays, and probability previews.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def create_discrepancy_map(s2_mask, l8_mask, out_path):
    """
    Creates an RGB spatial discrepancy map:
    - Green: Both agree on glacier
    - Blue: Sentinel-2 only
    - Red: Landsat only
    - Dark gray: Background
    """
    h, w = s2_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :] = [0.15, 0.15, 0.15]  # background

    both = (s2_mask == 1) & (l8_mask == 1)
    s2_only = (s2_mask == 1) & (l8_mask == 0)
    l8_only = (s2_mask == 0) & (l8_mask == 1)

    rgb[both] = [0.1, 0.85, 0.2]     # Green
    rgb[s2_only] = [0.1, 0.5, 0.95]   # Blue
    rgb[l8_only] = [0.95, 0.35, 0.1]  # Orange/Red

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title("Spatial Discrepancy Map: Sentinel-2 vs Landsat 9\n[Green: Agreement | Blue: S2-only | Orange: L8/9-only]", fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved spatial discrepancy map to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Visual Comparison Maps")
    parser.add_argument("--out-dir", type=str, default="results/comparison_maps")
    args = parser.parse_args()
    os.makedirs(os.path.join(repo_root, args.out_dir), exist_ok=True)
    print(f"[*] Visualizer ready for {args.out_dir}")


if __name__ == "__main__":
    main()
