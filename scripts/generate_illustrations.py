"""
Scientific Illustrations Generator for Glacier Segmentation Project
Generates high-resolution diagrams:
1. Attention U-Net Architecture Diagram
2. Attention Gate & CBAM Mechanism Diagram
3. Research Workflow Flowchart
4. Dual Satellite Cryospheric Monitoring Concept
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(repo_root, "presentation", "generated_illustrations")
os.makedirs(output_dir, exist_ok=True)


def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.96, "8-Channel Attention U-Net Architecture with CBAM Gates",
            ha="center", va="top", fontsize=20, fontweight="bold", color="#38BDF8")
    ax.text(0.5, 0.92, "Multispectral Input (B2, B3, B4, B8, B11, B12, NDSI, NDVI) -> Binary Glacier Segmentation",
            ha="center", va="top", fontsize=12, color="#94A3B8")

    # Encoder blocks (Left)
    encoder_levels = [
        ("Input\n(8×256×256)", 0.08, 0.78, 0.09, 0.10, "#3B82F6"),
        ("Conv Block 1\n(64, 256×256)", 0.19, 0.78, 0.11, 0.10, "#1E40AF"),
        ("Conv Block 2\n(128, 128×128)", 0.25, 0.62, 0.11, 0.10, "#1E40AF"),
        ("Conv Block 3\n(256, 64×64)", 0.31, 0.46, 0.11, 0.10, "#1E40AF"),
        ("Conv Block 4\n(512, 32×32)", 0.37, 0.30, 0.11, 0.10, "#1E40AF"),
    ]

    # Bottleneck
    bottleneck = ("Bottleneck Bridge\n(1024, 16×16)", 0.45, 0.14, 0.13, 0.10, "#7C3AED")

    # Decoder blocks (Right)
    decoder_levels = [
        ("UpConv 4 + AG\n(512, 32×32)", 0.55, 0.30, 0.11, 0.10, "#059669"),
        ("UpConv 3 + AG\n(256, 64×64)", 0.61, 0.46, 0.11, 0.10, "#059669"),
        ("UpConv 2 + AG\n(128, 128×128)", 0.67, 0.62, 0.11, 0.10, "#059669"),
        ("UpConv 1 + AG\n(64, 256×256)", 0.73, 0.78, 0.11, 0.10, "#059669"),
        ("Output Mask\n(1×256×256)", 0.86, 0.78, 0.09, 0.10, "#10B981"),
    ]

    # Attention Gates
    ag_boxes = [
        ("CBAM AG 4", 0.46, 0.30, 0.07, 0.07),
        ("CBAM AG 3", 0.46, 0.46, 0.07, 0.07),
        ("CBAM AG 2", 0.46, 0.62, 0.07, 0.07),
        ("CBAM AG 1", 0.46, 0.78, 0.07, 0.07),
    ]

    def draw_box(text, x, y, w, h, color, border="#38BDF8"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                              facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#F8FAFC")

    # Draw encoder
    for name, x, y, w, h, color in encoder_levels:
        draw_box(name, x, y, w, h, color)

    # Draw bottleneck
    draw_box(bottleneck[0], bottleneck[1], bottleneck[2], bottleneck[3], bottleneck[4], bottleneck[5], border="#A855F7")

    # Draw decoder
    for name, x, y, w, h, color in decoder_levels:
        draw_box(name, x, y, w, h, color, border="#34D399")

    # Draw AGs
    for name, x, y, w, h in ag_boxes:
        draw_box(name, x, y, w, h, "#B45309", border="#FBBF24")

    # Arrows - Encoder downward
    for i in range(len(encoder_levels) - 1):
        x1 = encoder_levels[i][1] + encoder_levels[i][3]/2
        y1 = encoder_levels[i][2]
        x2 = encoder_levels[i+1][1] + encoder_levels[i+1][3]/2
        y2 = encoder_levels[i+1][2] + encoder_levels[i+1][4]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#60A5FA", lw=2, mutation_scale=15))

    # Encoder to Bottleneck
    ax.annotate("", xy=(bottleneck[1] + bottleneck[3]/4, bottleneck[2] + bottleneck[4]),
                xytext=(encoder_levels[-1][1] + encoder_levels[-1][3]/2, encoder_levels[-1][2]),
                arrowprops=dict(arrowstyle="->", color="#60A5FA", lw=2, mutation_scale=15))

    # Bottleneck to Decoder
    ax.annotate("", xy=(decoder_levels[0][1] + decoder_levels[0][3]/4, decoder_levels[0][2]),
                xytext=(bottleneck[1] + 3*bottleneck[3]/4, bottleneck[2] + bottleneck[4]),
                arrowprops=dict(arrowstyle="->", color="#34D399", lw=2, mutation_scale=15))

    # Decoder upward
    for i in range(len(decoder_levels) - 2):
        x1 = decoder_levels[i][1] + decoder_levels[i][3]/2
        y1 = decoder_levels[i][2] + decoder_levels[i][4]
        x2 = decoder_levels[i+1][1] + decoder_levels[i+1][3]/2
        y2 = decoder_levels[i+1][2]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#34D399", lw=2, mutation_scale=15))

    # Decoder to output
    ax.annotate("", xy=(decoder_levels[-1][1], decoder_levels[-1][2] + decoder_levels[-1][4]/2),
                xytext=(decoder_levels[-2][1] + decoder_levels[-2][3], decoder_levels[-2][2] + decoder_levels[-2][4]/2),
                arrowprops=dict(arrowstyle="->", color="#10B981", lw=2.5, mutation_scale=18))

    # Skip connections via Attention Gates
    for i, (_, ag_x, ag_y, ag_w, ag_h) in enumerate(ag_boxes):
        enc_idx = 4 - i
        dec_idx = i
        # From encoder to AG
        x_enc = encoder_levels[enc_idx][1] + encoder_levels[enc_idx][3]
        y_enc = encoder_levels[enc_idx][2] + encoder_levels[enc_idx][4]/2
        ax.annotate("", xy=(ag_x, y_enc), xytext=(x_enc, y_enc),
                    arrowprops=dict(arrowstyle="->", color="#FBBF24", lw=1.8, linestyle="--", mutation_scale=12))

        # From AG to decoder
        x_dec = decoder_levels[dec_idx][1]
        y_dec = decoder_levels[dec_idx][2] + decoder_levels[dec_idx][4]/2
        ax.annotate("", xy=(x_dec, y_dec), xytext=(ag_x + ag_w, y_dec),
                    arrowprops=dict(arrowstyle="->", color="#FBBF24", lw=1.8, mutation_scale=12))

    # Legend / Key
    leg_x = 0.08
    leg_y = 0.05
    rect = FancyBboxPatch((leg_x, leg_y), 0.84, 0.07, boxstyle="round,pad=0.01",
                          facecolor="#1E293B", edgecolor="#475569", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(0.5, leg_y + 0.035,
            "Blue: Encoder Contracting Path | Violet: Latent Bridge | Amber: CBAM Attention Gates | Green: Expanding Path | Cyan: Skip Flow",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#E2E8F0")

    plt.tight_layout()
    out_file = os.path.join(output_dir, "attention_unet_architecture.png")
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[+] Saved Architecture Diagram to: {out_file}")


def generate_cbam_attention_gate_diagram():
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    ax.text(0.5, 0.96, "Convolutional Block Attention Module (CBAM) Skip Connection Gate",
            ha="center", va="top", fontsize=19, fontweight="bold", color="#FBBF24")
    ax.text(0.5, 0.91, "Channel Attention + Spatial Attention Gates Suppress Non-Cryospheric Noise & Shadow Artifacts",
            ha="center", va="top", fontsize=12, color="#94A3B8")

    def draw_rounded_card(x, y, w, h, title, subtitle, color, border):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                              facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.04, title, ha="center", va="top",
                fontsize=11, fontweight="bold", color="#F8FAFC")
        ax.text(x + w/2, y + h - 0.08, subtitle, ha="center", va="top",
                fontsize=8.5, color="#CBD5E1")

    # 1. Input Feature Map
    draw_rounded_card(0.05, 0.35, 0.14, 0.45, "Input Features (F)", "C × H × W\nFrom Encoder Skip", "#1E293B", "#38BDF8")

    # 2. Channel Attention Submodule
    draw_rounded_card(0.24, 0.22, 0.32, 0.58, "1. Channel Attention Module (Mc)", "Identifies 'WHAT' features (ice vs rock spectral response)", "#1E3A5F", "#60A5FA")
    ax.text(0.40, 0.62, "AvgPool(F) -> Shared MLP -> \\sigma\nMaxPool(F) -> Shared MLP /", ha="center", va="center",
            fontsize=10, color="#E2E8F0", family="monospace")
    ax.text(0.40, 0.48, "Mc(F) = \\sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#FBBF24", family="monospace")
    ax.text(0.40, 0.35, "F' = Mc(F) \\otimes F (Channel Refined)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#34D399", family="monospace")

    # 3. Spatial Attention Submodule
    draw_rounded_card(0.61, 0.22, 0.32, 0.58, "2. Spatial Attention Module (Ms)", "Identifies 'WHERE' features (glacier boundaries & crevasses)", "#1E3A5F", "#34D399")
    ax.text(0.77, 0.62, "AvgPool(F') along C -> [1×H×W]\nMaxPool(F') along C -> [1×H×W]", ha="center", va="center",
            fontsize=10, color="#E2E8F0", family="monospace")
    ax.text(0.77, 0.48, "Ms(F') = \\sigma(Conv7x7([AvgPool(F'); MaxPool(F')]))", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#FBBF24", family="monospace")
    ax.text(0.77, 0.35, "F'' = Ms(F') \\otimes F' (Dual Attended)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#34D399", family="monospace")

    # Arrows
    ax.annotate("", xy=(0.24, 0.52), xytext=(0.19, 0.52),
                arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2.5, mutation_scale=18))
    ax.annotate("", xy=(0.61, 0.52), xytext=(0.56, 0.52),
                arrowprops=dict(arrowstyle="->", color="#60A5FA", lw=2.5, mutation_scale=18))
    ax.annotate("", xy=(0.97, 0.52), xytext=(0.93, 0.52),
                arrowprops=dict(arrowstyle="->", color="#34D399", lw=2.5, mutation_scale=18))

    # Output text
    ax.text(0.98, 0.52, "To\nDecoder\nConcat", ha="left", va="center",
            fontsize=10, fontweight="bold", color="#10B981")

    # Bottom notes
    rect = FancyBboxPatch((0.05, 0.05), 0.90, 0.10, boxstyle="round,pad=0.01",
                          facecolor="#0B132B", edgecolor="#475569", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(0.5, 0.10,
            "Key Glaciological Benefit: Suppresses cloud cover, topographic cast shadows, and moraine debris\nwhile selectively amplifying glacial ice, firn, and clean glacier boundary gradients.",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#E2E8F0")

    plt.tight_layout()
    out_file = os.path.join(output_dir, "cbam_attention_gate_mechanism.png")
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[+] Saved CBAM Mechanism Diagram to: {out_file}")


def generate_research_workflow_diagram():
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    ax.text(0.5, 0.96, "End-to-End Autonomous Research Execution Workflow",
            ha="center", va="top", fontsize=19, fontweight="bold", color="#38BDF8")
    ax.text(0.5, 0.91, "From Satellite Acquisition to Attention U-Net Segmentation & Multi-Sensor Statistical Agreement",
            ha="center", va="top", fontsize=12, color="#94A3B8")

    stages = [
        ("Phase 1: Satellite Data Acquisition", "• AWS Earth Search STAC (S2 L2A)\n• Planetary Computer (Landsat 8/9)\n• RGI 6.0 Region 15 Outlines\n• Multi-year: 2021-2024", 0.04, "#1E3A8A"),
        ("Phase 2: NASA HLS Harmonization", "• EPSG:32644 (UTM 44N)\n• 30.0m Harmonized Grid\n• Bilinear band resampling\n• 1,466 Unique 256x256 Patches", 0.28, "#065F46"),
        ("Phase 3: Attention U-Net Training", "• 8 Multispectral Channels\n• CBAM Skip Gating Mechanism\n• Hybrid BCE-Dice Loss\n• Temporal Scene-Level Split", 0.52, "#581C87"),
        ("Phase 4: Statistical Evaluation", "• Paired t-test & Wilcoxon Test\n• 10,000 Bootstrap 95% CIs\n• Cohen's d & Cliff's delta\n• Bland-Altman Limits of Agreement", 0.76, "#831843"),
    ]

    for title, body, x, color in stages:
        rect = FancyBboxPatch((x, 0.22), 0.20, 0.60, boxstyle="round,pad=0.01",
                              facecolor=color, edgecolor="#38BDF8", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.10, 0.76, title, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color="#F8FAFC", wrap=True)
        ax.text(x + 0.02, 0.68, body, ha="left", va="top",
                fontsize=9.5, color="#E2E8F0", linespacing=1.6)

    # Arrows between phases
    for x_arr in [0.24, 0.48, 0.72]:
        ax.annotate("", xy=(x_arr + 0.04, 0.52), xytext=(x_arr, 0.52),
                    arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=3, mutation_scale=20))

    # Bottom deliverables bar
    rect = FancyBboxPatch((0.04, 0.06), 0.92, 0.10, boxstyle="round,pad=0.01",
                          facecolor="#1E293B", edgecolor="#10B981", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5, 0.11,
            "Research Deliverables: 1,466 Verified Patches • Best Checkpoint (.pth) • 24-Slide Presentation (.pptx) • Publication Figures & Reports",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#34D399")

    plt.tight_layout()
    out_file = os.path.join(output_dir, "research_workflow_flowchart.png")
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[+] Saved Research Workflow Flowchart to: {out_file}")


def generate_satellite_concept_diagram():
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    ax.text(0.5, 0.96, "Multi-Sensor Cryospheric Observation Architecture",
            ha="center", va="top", fontsize=19, fontweight="bold", color="#38BDF8")
    ax.text(0.5, 0.91, "Primary Sensor (Sentinel-2 MSI) vs Secondary Sensor (Landsat 8/9 OLI) over Gangotri Basin",
            ha="center", va="top", fontsize=12, color="#94A3B8")

    # Sentinel-2 Card
    rect1 = FancyBboxPatch((0.08, 0.22), 0.38, 0.60, boxstyle="round,pad=0.01",
                           facecolor="#1E3A8A", edgecolor="#38BDF8", linewidth=1.8)
    ax.add_patch(rect1)
    ax.text(0.27, 0.77, "Primary Satellite: Sentinel-2 MSI", ha="center", va="top",
            fontsize=13, fontweight="bold", color="#38BDF8")
    ax.text(0.11, 0.69,
            "• Agency: European Space Agency (Copernicus)\n"
            "• Resolution: 10m (VNIR), 20m (SWIR)\n"
            "• Revisit Rate: 5 days (Constellation A+B)\n"
            "• Bands: B2 (Blue), B3 (Green), B4 (Red),\n"
            "         B8 (NIR), B11 (SWIR1), B12 (SWIR2)\n"
            "• Processing Level: Level-2A BOA Reflectance\n"
            "• Role: Core Attention U-Net training sensor\n"
            "• Status: 1,466 Patches Extracted & Verified",
            ha="left", va="top", fontsize=10, color="#E2E8F0", linespacing=1.6)

    # Landsat 8/9 Card
    rect2 = FancyBboxPatch((0.54, 0.22), 0.38, 0.60, boxstyle="round,pad=0.01",
                           facecolor="#831843", edgecolor="#F43F5E", linewidth=1.8)
    ax.add_patch(rect2)
    ax.text(0.73, 0.77, "Secondary Satellite: Landsat 8/9 OLI", ha="center", va="top",
            fontsize=13, fontweight="bold", color="#F43F5E")
    ax.text(0.57, 0.69,
            "• Agency: USGS / NASA\n"
            "• Resolution: 30m (VNIR/SWIR), 15m (Panchromatic)\n"
            "• Revisit Rate: 8 days (Constellation 8+9)\n"
            "• Bands: B2 (Blue), B3 (Green), B4 (Red),\n"
            "         B5 (NIR), B6 (SWIR1), B7 (SWIR2)\n"
            "• Processing Level: Collection 2 Level-2 Surface Reflectance\n"
            "• Role: Independent statistical comparison\n"
            "• Status: 756 Paired 30m Patches Evaluated",
            ha="left", va="top", fontsize=10, color="#E2E8F0", linespacing=1.6)

    # Central comparison arrow
    ax.annotate("", xy=(0.54, 0.52), xytext=(0.46, 0.52),
                arrowprops=dict(arrowstyle="<->", color="#FBBF24", lw=3, mutation_scale=20))
    ax.text(0.50, 0.56, "HLS 30m\nCross-Sensor\nValidation", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#FBBF24")

    # Bottom summary
    rect3 = FancyBboxPatch((0.08, 0.06), 0.84, 0.10, boxstyle="round,pad=0.01",
                           facecolor="#0F172A", edgecolor="#475569", linewidth=1.2)
    ax.add_patch(rect3)
    ax.text(0.5, 0.11,
            "Spatial Alignment: Harmonized to common 30.0m UTM Zone 44N (EPSG:32644) grid with RGI 6.0 ground truth.",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#E2E8F0")

    plt.tight_layout()
    out_file = os.path.join(output_dir, "satellite_comparison_concept.png")
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[+] Saved Satellite Comparison Concept to: {out_file}")


def main():
    print("=" * 60)
    print("🎨 GENERATING RESEARCH ILLUSTRATIONS")
    print("=" * 60)
    generate_architecture_diagram()
    generate_cbam_attention_gate_diagram()
    generate_research_workflow_diagram()
    generate_satellite_concept_diagram()
    print("[+] All 4 illustrations generated successfully.")


if __name__ == "__main__":
    main()
