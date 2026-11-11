import os
import sys
import argparse
import numpy as np

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from models.attention_unet import AttentionUNet
from src.preprocessing import compute_spectral_indices, normalize_channels


def main():
    parser = argparse.ArgumentParser(description="Full-Scene / Patch Inference with Attention U-Net")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Model weights")
    parser.add_argument("--input", type=str, default=None, help="Path to 8-channel or 6-channel .npy patch")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification threshold")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = AttentionUNet(in_channels=8, out_channels=1).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[+] Loaded weights from {args.checkpoint}")
    else:
        print(f"[!] Warning: Checkpoint {args.checkpoint} not found. Running with uninitialized model.")

    model.eval()

    # Load input patch
    input_path = args.input
    if input_path is None:
        default_dir = "data/patches"
        candidates = sorted([os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith("_patch.npy")])
        if candidates:
            # Pick a patch with ice
            input_path = candidates[0]
        else:
            raise FileNotFoundError("No input patch specified and 'data/patches' is empty.")

    print(f"[*] Running inference on genuine satellite patch: {input_path}")
    patch = np.load(input_path)

    if patch.shape[0] == 6:
        patch = compute_spectral_indices(patch)
    patch_norm = normalize_channels(patch)

    tensor = torch.from_numpy(patch_norm).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (probs >= args.threshold).astype(np.uint8)

    # Plot visual prediction preview
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # False-color RGB (NIR: B8, Red: B4, Green: B3)
    rgb = np.stack([patch[3], patch[2], patch[1]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    axes[0].imshow(rgb)
    axes[0].set_title(f"Input False Color (NIR-R-G)\n[{os.path.basename(input_path)}]", fontweight="bold")
    axes[0].axis('off')

    axes[1].imshow(probs, cmap='magma', vmin=0.0, vmax=1.0)
    axes[1].set_title("Glacier Probability Map", fontweight="bold")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='Blues')
    axes[2].set_title(f"Predicted Binary Mask (Threshold={args.threshold:.2f})\nIce Pixels: {np.sum(pred_mask):,}", fontweight="bold")
    axes[2].axis('off')

    plt.tight_layout()
    out_preview = os.path.join(args.output_dir, "prediction_preview.png")
    plt.savefig(out_preview, dpi=300)
    plt.close()

    mask_out = os.path.join(args.output_dir, "predicted_mask.npy")
    np.save(mask_out, pred_mask)

    print(f"[+] Exported prediction visualization to: {out_preview}")
    print(f"[+] Exported binary prediction mask to: {mask_out}")


if __name__ == "__main__":
    main()
