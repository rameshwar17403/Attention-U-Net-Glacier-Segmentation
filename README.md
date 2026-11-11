# 🧊 Attention U-Net Multispectral Glacier Segmentation Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Verified Real Data](https://img.shields.io/badge/Status-Verified%20Real%20Data-brightgreen.svg)]()

An end-to-end, scientifically validated Deep Learning and Remote Sensing framework implementing an **8-Channel Attention U-Net** enhanced with **CBAM (Convolutional Block Attention Modules)** and **Focal Tversky Loss** for automated, high-precision glacier segmentation. The pipeline evaluates genuine multispectral satellite imagery (**Copernicus Sentinel-2 MSI Level-2A** and **USGS/NASA Landsat 9 OLI-2 Level-2**) benchmarked against official **Randolph Glacier Inventory (RGI 6.0)** ground truth vector outlines over the Gangotri Glacier basin, Garhwal Himalaya, India.

- **Academic Context**: B.Tech Information Technology (7th Semester Minor Project), National Institute of Technology (NIT) Raipur.
- **Authors**: Piyush Verma, Deepanshu Sahu, Rameshwar Chandravanshi.

---

## 📑 Table of Contents

1. [Project Overview & Scientific Objectives](#-1-project-overview--scientific-objectives)
2. [Study Area: Gangotri Glacier Basin](#-2-study-area-gangotri-glacier-basin)
3. [Genuine Satellite & Reference Data Sources](#-3-genuine-satellite--reference-data-sources)
4. [The 8-Channel Multispectral Tensor Pipeline](#-4-the-8-channel-multispectral-tensor-pipeline)
5. [Neural Architecture & CBAM Attention Mechanism](#-5-neural-architecture--cbam-attention-mechanism)
6. [Loss Formulations (Hybrid BCE-Dice & Focal Tversky)](#-6-loss-formulations-hybrid-bce-dice--focal-tversky)
7. [Automated Data Acquisition & Preprocessing Pipeline](#-7-automated-data-acquisition--preprocessing-pipeline)
8. [Training & Evaluation Methodology](#-8-training--evaluation-methodology)
9. [Multi-Satellite Statistical Comparison Results](#-9-multi-satellite-statistical-comparison-results)
10. [Repository Structure](#-10-repository-structure)
11. [Installation & Local Setup](#-11-installation--local-setup)
12. [Execution Commands (End-to-End Reproduction)](#-12-execution-commands-end-to-end-reproduction)
13. [Implementation Status & Audit Summary](#-13-implementation-status--audit-summary)
14. [Limitations & Scientific Discussion](#-14-limitations--scientific-discussion)
15. [References & Data Attribution](#-15-references--data-attribution)
16. [License](#-16-license)

---

## 🔭 1. Project Overview & Scientific Objectives

Accurate delineation of alpine glaciers is critical for assessing cryospheric freshwater reserves, monitoring climate-induced retreat, and mitigating Glacial Lake Outburst Flood (GLOF) hazards. In rugged Himalayan terrain, manual delineation is subjective and labor-intensive, while conventional thresholding methods (such as simple NDSI cutoff) fail over supraglacial debris cover and cast terrain shadows.

### Research Objectives:
1. **Automated Multispectral Segmentation**: Ingest 6 surface reflectance bands plus calculated physical indices (NDSI and NDVI) into an Attention U-Net with CBAM attention gates.
2. **Debris-Covered Ice Delineation**: Optimize segmentation using Focal Tversky Loss to heavily penalize false negatives along elusive debris-covered glacier margins.
3. **Cross-Sensor Generalization & Statistical Rigor**: Formally compare Sentinel-2 MSI (10m) against Landsat 9 OLI (30m) on identical spatial footprints using non-parametric hypothesis testing (Wilcoxon signed-rank), 10,000-sample bootstrap confidence intervals, and Bland-Altman agreement analysis.

---

## 🏔️ 2. Study Area: Gangotri Glacier Basin

- **Geographical Coordinates**: $30.80^\circ\text{N} - 31.05^\circ\text{N}$, $79.05^\circ\text{E} - 79.30^\circ\text{E}$
- **Geographic Location**: Uttarkashi District, Uttarakhand, India (Garhwal Himalaya).
- **Glaciological Importance**: Gangotri is one of the largest valley glaciers in the Himalayas (~30 km length, >120 km² area), feeding the Bhagirathi River (a primary headstream of the Ganges). It features clean accumulation ice, heavily debris-mantled ablation zones, and extensive lateral moraines.
- **Projected Coordinate Reference System**: WGS 84 / UTM Zone 44N (`EPSG:32644`).

---

## 🛰️ 3. Genuine Satellite & Reference Data Sources

All experiments use authentic, unsimulated satellite observations acquired during the ablation minimum season (October 2024):

| Dataset | Platform / Sensor | Product Level | Scene Identifier | Acquisition Date | Cloud Cover | Source Provider |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentinel-2** | Sentinel-2B MSI | Level-2A (Surface Reflectance) | `S2B_44RLV_20241015_0_L2A` | Oct 15, 2024 | 0.02% | AWS Earth Search COG Archive |
| **Landsat 9** | Landsat 9 OLI-2 | Level-2 (Surface Reflectance) | `LC09_L2SP_146039_20241026_02_T1` | Oct 26, 2024 | 1.96% | Microsoft Planetary Computer STAC |
| **Glacier Labels** | Randolph Glacier Inventory | RGI 6.0 Region 15 (South Asia East) | 100 Basin Polygons (`RGI60-15.06881` Gangotri) | Validated Reference | N/A | OGGM / GLIMS Global Inventory |

---

## 🧪 4. The 8-Channel Multispectral Tensor Pipeline

Every pixel is represented as an 8-dimensional cryospheric feature vector:

$$\mathbf{X} = [B_2, B_3, B_4, B_8, B_{11}, B_{12}, \text{NDSI}_{\text{norm}}, \text{NDVI}_{\text{norm}}] \in \mathbb{R}^{8 \times H \times W}$$

1. **Band 2 (Blue - 490 nm)**: Water penetration, atmospheric scattering correction.
2. **Band 3 (Green - 560 nm)**: High clean ice reflectance, base band for NDSI.
3. **Band 4 (Red - 665 nm)**: Separates barren bedrock and alpine vegetation from ice.
4. **Band 8 (NIR - 842 nm)**: Strong absorption by water and ice bodies; vegetation contrast.
5. **Band 11 (SWIR-1 - 1610 nm)**: Fundamental cryospheric discriminator; snow and ice exhibit near-total absorption, whereas clouds remain highly reflective.
6. **Band 12 (SWIR-2 - 2190 nm)**: Supraglacial moraine and mineralogical discrimination.
7. **NDSI (Normalized Difference Snow Index)**:
   $$\text{NDSI} = \frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1} + \epsilon} = \frac{B_3 - B_{11}}{B_3 + B_{11} + \epsilon}$$
8. **NDVI (Normalized Difference Vegetation Index)**:
   $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + \epsilon} = \frac{B_8 - B_4}{B_8 + B_4 + \epsilon}$$

Spectral indices are scaled from $[-1, +1]$ to $[0, 1]$ via $\text{Index}_{\text{norm}} = \frac{\text{Index} + 1.0}{2.0}$.

---

## 🧠 5. Neural Architecture & CBAM Attention Mechanism

The architecture is an enhanced **Attention U-Net** featuring:
- **Encoder**: 4-stage contracting path with double $3\times 3$ convolutions, BatchNorm, ReLU, and $2\times 2$ Max Pooling (`[64, 128, 256, 512, 1024]` features).
- **Decoder**: 4-stage expanding path using $2\times 2$ transposed convolutions concatenated with attention-filtered skip features.
- **CBAM Attention Gate**: Skip connections from the encoder are modulated by gating signal $\mathbf{g}$ from the lower decoder stage and refined by sequential **Channel Attention** and **Spatial Attention**:

$$\mathbf{M}_c(\mathbf{F}) = \sigma(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F})))$$

$$\mathbf{M}_s(\mathbf{F}) = \sigma(f^{7\times 7}([\text{AvgPool}(\mathbf{F}); \text{MaxPool}(\mathbf{F})]))$$

$$\mathbf{F}_{\text{out}} = \text{CBAM}(\mathbf{x} \cdot \sigma(\mathbf{W}_g \mathbf{g} + \mathbf{W}_x \mathbf{x}))$$

---

## ⚖️ 6. Loss Formulations (Hybrid BCE-Dice & Focal Tversky)

1. **Hybrid BCE-Dice Loss**:
   $$\mathcal{L}_{\text{Hybrid}} = \alpha \cdot \mathcal{L}_{\text{BCE}}(p, y) + (1 - \alpha) \cdot \mathcal{L}_{\text{Dice}}(p, y)$$

2. **Focal Tversky Loss** (Addresses class imbalance and debris-covered ice omission):
   $$\text{TI} = \frac{\text{TP} + \epsilon}{\text{TP} + \alpha \cdot \text{FP} + \beta \cdot \text{FN} + \epsilon}$$
   $$\mathcal{L}_{\text{FTL}} = (1 - \text{TI})^\gamma$$
   _(Hyperparameters: $\alpha = 0.3$, $\beta = 0.7$ to prioritize recall over precision, $\gamma = 1.33$)._

---

## 🛰️ 7. Automated Data Acquisition & Preprocessing Pipeline

Run `scripts/acquire_satellite_data.py` to reproduce the exact dataset:
- Queries STAC endpoints (AWS Element84 and Planetary Computer).
- Performs windowed Cloud-Optimized GeoTIFF (COG) reads directly over HTTPS without local bulk downloads.
- Aligns both sensors onto an identical 10m UTM Zone 44N grid ($2434 \times 2810$ pixels).
- Downloads RGI 6.0 Region 15 shapefile and rasterizes the 100 glacier polygons covering Gangotri basin ($2,741,517$ glacier pixels, $40.08\%$ of basin).
- Generates 32 clean non-overlapping $256 \times 256$ cryospheric patches and masks.
- Exports metadata audit to `data/satellite_patches/acquisition_metadata.json`.

---

## 📈 8. Training & Evaluation Methodology

The model was trained on genuine Sentinel-2 Level-2A patches with RGI 6.0 masks using Apple Silicon Metal (MPS) GPU acceleration:
- **Optimizer**: AdamW ($\text{lr} = 0.0005$, $\text{weight\_decay} = 0.0001$)
- **Batch Size**: 4 | **Epochs**: 15
- **Convergence**:
  - Best Validation Dice: **0.8100** (Epoch 8)
  - Validation Loss: **0.2993** (Epoch 14)
  - Overall Test Dice: **0.6925** | Overall IoU: **0.5297**
  - Confusion Matrix: $\text{TP} = 413,022$, $\text{FP} = 213,058$, $\text{FN} = 153,695$, $\text{TN} = 1,317,377$

---

## 📊 9. Multi-Satellite Statistical Comparison Results

Using `scripts/statistical_sensor_analysis.py`, 32 matched patches were evaluated under the cross-sensor transfer protocol (Sentinel-2 trained Attention U-Net evaluated on Sentinel-2 10m vs. Landsat 9 30m):

### 9.1 Metric Distributions (32 Matched Patches)

| Metric | Sentinel-2 MSI (10m) Mean ± SD | Sentinel-2 Median [IQR] | Landsat 9 OLI (30m) Mean ± SD | Landsat 9 Median [IQR] | Mean Difference (S2 − L8) | 95% Bootstrap CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dice Score** | **0.5037 ± 0.337** | **0.6280** [0.730] | 0.4645 ± 0.336 | 0.5627 [0.722] | **+0.0391** (+3.91%) | $[-0.0220, +0.1000]$ |
| **IoU Score** | **0.4000 ± 0.294** | **0.4578** [0.609] | 0.3639 ± 0.293 | 0.3915 [0.589] | **+0.0360** (+3.60%) | $[-0.0234, +0.0917]$ |
| **Precision** | 0.5255 ± 0.341 | 0.6694 [0.672] | 0.5063 ± 0.352 | 0.6573 [0.716] | +0.0192 (+1.92%) | $[-0.0401, +0.0695]$ |
| **Recall** | **0.5759 ± 0.388** | **0.7400** [0.815] | 0.5213 ± 0.390 | 0.6536 [0.803] | **+0.0545** (+5.45%) | $[-0.0248, +0.1370]$ |

### 9.2 Hypothesis Testing & Agreement Analysis

| Test Name | Statistic | $p$-value | Significant ($\alpha=0.05$)? | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Shapiro-Wilk Test** | $W = 0.771$ | $p = 3.04 \times 10^{-5}$ | Non-Normal | Metric differences violate normality; non-parametric tests are required. |
| **Wilcoxon Signed-Rank** | $W = 100.0$ | **$p = 0.0325$** | **Yes ($p < 0.05$)** | **Sentinel-2 exhibits statistically significant superior Dice over Landsat 9.** |
| **Paired Student's t-test** | $t = 1.244$ | $p = 0.2229$ | No | Parametric test invalid due to distribution skewness. |
| **Mann-Whitney U Test** | $U = 538.0$ | $p = 0.7312$ | No | Demonstrates why pairing is necessary; independent test loses spatial matching power. |
| **Cohen's $d$ Effect Size** | $d = 0.220$ | N/A | Small Positive | Quantifies positive advantage of 10m spatial resolution. |
| **Bland-Altman Agreement** | Bias = +0.039 | LoA: $[-0.309, +0.387]$ | 93.75% in LoA | High sensor concordance with systematic bias favoring Sentinel-2 along boundaries. |

---

## 📁 10. Repository Structure

```
Attention-U-Net-Glacier-Segmentation/
├── README.md                      # Comprehensive scientific documentation
├── PROJECT_AUDIT.md               # Full audit report of code, weights & data
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git exclusion manifest
├── LICENSE                        # MIT License
│
├── configs/
│   └── default.yaml               # Hyperparameters, paths & thresholds
│
├── models/
│   ├── attention_unet.py          # 8-Channel Attention U-Net with CBAM gates
│   └── cbam.py                    # Standalone Channel & Spatial Attention module
│
├── src/
│   ├── dataset.py                 # PyTorch Dataset for 8-channel cryospheric patches
│   ├── engine.py                  # Training and validation epoch loops
│   ├── losses.py                  # Hybrid BCE-Dice, Tversky & Focal Tversky Loss
│   ├── metrics.py                 # Dice, IoU, Precision, Recall calculations
│   ├── preprocessing.py           # Min-max scaling and NDSI/NDVI computation
│   └── utils.py                   # Seeding utilities and robust YAML loader
│
├── scripts/
│   ├── acquire_satellite_data.py  # STAC downloader for S2, L8, and RGI 6.0
│   ├── train.py                   # Training pipeline with history logging
│   ├── evaluate.py                # Model evaluation & confusion matrix generator
│   ├── predict.py                 # Full patch inference & probability export
│   ├── statistical_sensor_analysis.py # Multi-satellite statistical analysis suite
│   ├── inspect_geotiff.py         # Raster inspection utility
│   └── prepare_dataset.py         # Mock data generator (smoke testing only)
│
├── tests/
│   ├── test_dataset.py            # Unit tests for patch loading
│   ├── test_losses.py             # Unit tests for loss functions
│   ├── test_metrics.py            # Unit tests for segmentation metrics
│   └── test_model.py              # Unit tests for AttentionUNet & CBAM
│
├── outputs/
│   ├── checkpoints/               # Trained model weights (best_model.pth)
│   ├── evaluation/                # Training curves, history CSV & overlay preview
│   ├── predictions/               # Inference masks and probability previews
│   └── statistical_analysis/      # CSV tables, reports & 5 publication figures
│       └── figures/
│           ├── sensor_comparison_summary.png
│           ├── dice_distribution.png
│           ├── iou_comparison.png
│           ├── confidence_intervals.png
│           └── prediction_comparison.png
└── data/
    ├── README.md                  # Dataset guide and patch format specifications
    └── patches/                   # Genuine Sentinel-2 patches & RGI masks
```

---

## ⚙️ 11. Installation & Local Setup

```bash
# 1. Clone repository
git clone https://github.com/rameshwar17403/Attention-U-Net-Glacier-Segmentation.git
cd Attention-U-Net-Glacier-Segmentation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# .\venv\Scripts\Activate.ps1  # On Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 12. Execution Commands (End-to-End Reproduction)

```bash
# Run unit test suite (9 tests passing)
python -m unittest discover tests

# 1. Acquire genuine Sentinel-2, Landsat 9, and RGI 6.0 patches
python scripts/acquire_satellite_data.py \
    --bbox 79.05 30.80 79.30 31.05 \
    --date-range 2024-09-01T00:00:00Z/2024-10-31T23:59:59Z \
    --output-dir data/satellite_patches

# 2. Train Attention U-Net on genuine Sentinel-2 patches
python scripts/train.py --epochs 15 --batch-size 4 --lr 0.0005

# 3. Evaluate trained checkpoint
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --threshold 0.50

# 4. Run single-patch inference & visual overlay
python scripts/predict.py --checkpoint outputs/checkpoints/best_model.pth --threshold 0.50

# 5. Execute multi-satellite statistical analysis & generate figures
python scripts/statistical_sensor_analysis.py \
    --s2-dir data/satellite_patches/s2_patches \
    --landsat-dir data/satellite_patches/landsat_patches \
    --mask-dir data/satellite_patches/masks \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output-dir outputs/statistical_analysis \
    --threshold 0.50
```

---

## 📋 13. Implementation Status & Audit Summary

| Component | Status | Verification Detail |
| :--- | :--- | :--- |
| **Attention U-Net + CBAM** | ✅ Implemented & Tested | Validated in `tests/test_model.py` (9 tests passing) |
| **Focal Tversky Loss** | ✅ Implemented & Tested | Validated in `tests/test_losses.py` |
| **Genuine Sentinel-2 Acquisition** | ✅ Implemented & Tested | S2B L2A (Oct 15, 2024) windowed COG read via AWS Earth Search |
| **Genuine Landsat 9 Acquisition** | ✅ Implemented & Tested | LC09 L2 (Oct 26, 2024) via Planetary Computer STAC |
| **RGI 6.0 Ground Truth Masking** | ✅ Implemented & Tested | 100 Gangotri basin polygons rasterized at 10m UTM 44N |
| **Model Training on Real Data** | ✅ Implemented & Executed | 15 epochs on MPS GPU; Best Val Dice = 0.8100 |
| **Cross-Sensor Statistical Engine** | ✅ Implemented & Executed | Real inference on 32 patches; Wilcoxon $p = 0.0325$ |
| **Full Regional Transfer (e.g. Karakoram / Alps)** | 🔄 Planned | Documented for future multi-basin generalization |

---

## ⚠️ 14. Limitations & Scientific Discussion

1. **Debris-Covered Ice Complexity**: Lateral moraines and thick supraglacial debris mimic adjacent valley rock signatures. While Focal Tversky Loss improves boundary recall, multi-temporal thermal infrared (TIR) or SAR coherence data would further refine debris-covered margins.
2. **Topographic Shadows**: Deep north-facing cast shadows in steep Himalayan cirques attenuate optical reflectance.
3. **Sensor Resolution Contrast**: Sentinel-2 (10m) demonstrates a statistically significant +3.91% Dice advantage over Landsat 9 (30m), primarily due to edge boundary pixel mixing at 30m resolution.

---

## 📚 15. References & Data Attribution

- **Sentinel-2 Data**: Copernicus Programme, European Space Agency (ESA) accessed via AWS Open Data.
- **Landsat Data**: U.S. Geological Survey (USGS) and NASA via Microsoft Planetary Computer STAC.
- **Glacier Reference Outlines**: Randolph Glacier Inventory (RGI Consortium, 2017), RGI 6.0 Region 15 (South Asia East) via GLIMS and OGGM.
- **Attention U-Net**: O. Oktay et al., *“Attention U-Net: Learning Where to Look for the Pancreas”*, arXiv:1804.03999, 2018.
- **CBAM**: S. Woo et al., *“CBAM: Convolutional Block Attention Module”*, ECCV 2018.

---

## 📜 16. License

This project is licensed under the [MIT License](LICENSE).
