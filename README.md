# 🧊 Attention U-Net Multispectral Glacier Segmentation Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform: Windows | Linux | Colab](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Colab-green.svg)]()

An independent, production-grade Deep Learning framework implementing an **8-Channel Attention U-Net** enhanced with **CBAM (Convolutional Block Attention Modules)** and **Focal Tversky Loss** for automated, high-precision glacier and glacial lake segmentation in rugged alpine terrain using Copernicus Sentinel-2 multispectral imagery and Randolph Glacier Inventory (RGI) reference labels.

- **Academic Context**: B.Tech Information Technology (7th Semester Minor Project), National Institute of Technology (NIT) Raipur.
- **Authors**: Piyush Verma, Deepanshu Sahu, Rameshwar Chandravanshi.

---

## 📑 Table of Contents

1. [Project Overview & Motivation](#-project-overview--motivation)
2. [Problem Statement & Scientific Challenges](#-problem-statement--scientific-challenges)
3. [Version Evolution & Changelog (v1.0 to v2.0)](#-version-evolution--changelog-v10-to-v20)
4. [The 8-Channel Multispectral Input Pipeline](#-the-8-channel-multispectral-input-pipeline)
5. [Attention U-Net Architecture & CBAM Integration](#-attention-u-net-architecture--cbam-integration)
6. [Advanced Loss Functions (Hybrid BCE-Dice & Focal Tversky)](#-advanced-loss-functions-hybrid-bce-dice--focal-tversky)
7. [Statistical Sensor & Dataset Analysis Framework](#-statistical-sensor--dataset-analysis-framework)
8. [Evaluation Metrics & Threshold Tuning](#-evaluation-metrics--threshold-tuning)
9. [Repository Structure](#-repository-structure)
10. [Installation & Local Setup](#-installation--local-setup)
11. [Running Tests, Training, Evaluation & Inference](#-running-tests-training-evaluation--inference)
12. [Google Colab Workflow](#-google-colab-workflow)
13. [Data Attribution & References](#-data-attribution--references)
14. [License](#-license)

---

## 🔭 1. Project Overview & Motivation

Manual delineation of glaciers and glacial lakes from satellite imagery is time-consuming, subjective, and practically infeasible at regional scales due to the sheer volume of high-resolution remote-sensing data. Climate change accelerates glacial retreat and expands dangerous glacial lake outburst flood (GLOF) hazards in high mountain ranges such as the Himalayas.

This project builds a fully automated computer vision and remote-sensing pipeline that ingests orbital satellite observations, computes physical cryospheric indices, and uses a state-of-the-art **Attention U-Net** to segment ice bodies with pixel-level precision.

---

## 🏔️ 2. Problem Statement & Scientific Challenges

Standard deep learning segmentation models trained on natural 3-channel RGB imagery struggle in high-altitude terrain due to:

- **Debris-Covered Ice**: Supraglacial debris mimics the spectral signature of surrounding moraines and barren rock walls.
- **Topographic Shadows**: Deep mountain shadows obscure clean ice boundaries in standard optical bands.
- **Seasonal Snow Noise**: Transient seasonal snow packs confuse basic thresholding algorithms.

**Our Solution**: We leverage an **8-channel tensor** combining visible/infrared surface reflectance bands with physics-based spectral indices (`NDSI` and `NDVI`), processed through **Attention Gates (AGs)** and **CBAM** that dynamically suppress background clutter and isolate true glacier margins.

---

## 📈 3. Version Evolution & Changelog (v1.0 to v2.0)

### Version 2.0 (Current - Advanced Upgrades & Statistical Rigor)

- **CBAM (Convolutional Block Attention Module)**: Upgraded standard spatial-only Attention Gates into dual **Channel and Spatial Attention** modules. Channel attention dynamically weights _what_ spectral bands matter (SWIR, NIR, NDSI, NDVI), while spatial attention isolates _where_ glacier boundaries lie.
- **Advanced Loss Formulation**: Introduced **Focal Tversky Loss** alongside Hybrid BCE-Dice Loss to heavily penalize false negatives and successfully capture elusive debris-covered glacier ice.
- **Real Multi-Satellite Statistical Analysis Framework**: Added `scripts/statistical_sensor_analysis.py` to execute real patch-level Attention U-Net inference, paired t-tests, Wilcoxon signed-rank tests, Mann-Whitney U tests, Bland-Altman agreement, and 10,000-iteration bootstrap 95% confidence intervals comparing Sentinel-2 MSI and Landsat 8/9 OLI / HLS.
- **Automated Data Preparation**: Added `scripts/acquire_satellite_data.py` for cloud-optimized multi-satellite data acquisition and patch extraction.
- **Stability & Refactoring**: Resolved tensor channel dimension alignments, added automatic fallback synthetic dataset generation, and optimized execution across CPU and CUDA backends.

### Version 1.0 (Initial Baseline Implementation)

- Standard 8-channel input pipeline (6 Sentinel-2 L2A surface reflectance bands + NDSI + NDVI).
- Standard Attention U-Net using additive Attention Gates.
- Hybrid Binary Cross-Entropy + Soft Dice loss optimization.

---

## 🛰️ 4. The 8-Channel Multispectral Input Pipeline

To maximize discriminatory power across snow, firn, clean ice, supraglacial debris, and barren bedrock, each pixel is represented across eight dedicated dimensions:

1. **Band 2 (Blue - 490 nm)**: Sensitive to atmospheric haze and deep clean water bodies.
2. **Band 3 (Green - 560 nm)**: Critical component for snow index calculation.
3. **Band 4 (Red - 665 nm)**: Differentiates bare soil and sparse alpine vegetation from ice.
4. **Band 8 (NIR - 842 nm)**: Strong absorption by water/ice and high reflection by vegetation.
5. **Band 11 (SWIR-1 - 1610 nm)**: Critical cryospheric discriminator; snow and ice exhibit strong absorption, appearing exceptionally dark, whereas clouds remain highly reflective.
6. **Band 12 (SWIR-2 - 2190 nm)**: Resolves mineralogical compositions and supraglacial moraines.
7. **NDSI (Normalized Difference Snow Index)**:
   $$\text{NDSI} = \frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1} + \epsilon} = \frac{\text{B03} - \text{B11}}{\text{B03} + \text{B11} + \epsilon}$$
8. **NDVI (Normalized Difference Vegetation Index)**:
   $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + \epsilon} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04} + \epsilon}$$

Both physical indices are normalized from $[-1, +1]$ to $[0, 1]$ before concatenation:

$$\text{Index}_{\text{norm}} = \frac{\text{Index} + 1.0}{2.0}$$

---

## 🧠 5. Attention U-Net Architecture & CBAM Integration

Standard U-Nets concatenate encoder feature maps directly to the decoder via skip connections, indiscriminately transferring spatial noise and background clutter (rock shadows, non-glacial water bodies).

We integrate **Attention Gates (AGs)** combined with **CBAM (Convolutional Block Attention Modules)**:

1. **Channel Attention (Spectral Dimension)**: Squeezes spatial dimensions via Adaptive Average and Max Pooling, feeding into a shared MLP to highlight salient bands:
   $$\mathbf{M}_c(\mathbf{F}) = \sigma(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F})))$$
2. **Spatial Attention (Boundary Isolation)**: Convolves concatenated spatial channel projections with a $7\times 7$ kernel to emphasize glacier outlines and terminus margins:
   $$\mathbf{M}_s(\mathbf{F}) = \sigma(f^{7\times 7}([\text{AvgPool}(\mathbf{F}); \text{MaxPool}(\mathbf{F})]))$$
3. **Additive Attention Gate**: Combines decoder gating signal $\mathbf{g}$ with skip feature $\mathbf{x}$ through $1\times 1$ convolutions followed by ReLU and Sigmoid activations.

---

## ⚖️ 6. Advanced Loss Functions (Hybrid BCE-Dice & Focal Tversky)

To handle severe class imbalance, we implement two advanced loss modules in `src/losses.py`:

1. **Hybrid BCE-Dice Loss**:
   $$\mathcal{L}_{\text{Hybrid}} = \alpha \cdot \mathcal{L}_{\text{BCE}}(logits, targets) + (1 - \alpha) \cdot \mathcal{L}_{\text{Dice}}(logits, targets)$$

2. **Focal Tversky Loss**:
   $$\text{TI} = \frac{\text{TP} + \epsilon}{\text{TP} + \alpha \cdot \text{FP} + \beta \cdot \text{FN} + \epsilon}$$
   $$\mathcal{L}_{\text{FocalTversky}} = (1 - \text{TI})^\gamma$$
   _(where weighting parameters $\alpha = 0.3$, $\beta = 0.7$, and $\gamma = 1.33$ heavily penalize false negatives to capture debris-covered ice margins)._

---

## 📊 7. Statistical Sensor & Dataset Analysis Framework

To rigorously answer comparative research questions between sensor datasets (e.g., Copernicus Sentinel-2 MSI vs. USGS/NASA Landsat 8/9 OLI under the NASA Harmonized Landsat-Sentinel HLS v2 framework), our framework implements statistical evaluation via `scripts/statistical_sensor_analysis.py`:

- **Parametric & Non-Parametric Descriptives**: Mean, Standard Deviation, Median, IQR, Min, Max.
- **Normality Assessment**: Shapiro-Wilk test on paired score differences.
- **Paired Hypothesis Testing**: Paired Student's t-test and Wilcoxon Signed-Rank Test.
- **Independent Comparison**: Mann-Whitney U Test.
- **Bootstrapped 95% Confidence Intervals**: 10,000 iterations to quantify estimation precision.
- **Effect Size Estimation**: Cohen's $d$ and Cliff's $\delta$.
- **Bland-Altman Agreement**: Mean bias and 95% Limits of Agreement.

---

## 📏 8. Evaluation Metrics & Threshold Tuning

1. **Dice Coefficient (F1-Score)**: $\frac{2 |P \cap G|}{|P| + |G| + \epsilon}$
2. **Intersection over Union (IoU)**: $\frac{|P \cap G|}{|P \cup G| + \epsilon}$
3. **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP} + \epsilon}$
4. **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN} + \epsilon}$

---

## 📁 9. Repository Structure

```
Attention-U-Net-Glacier-Segmentation/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Pip dependency manifest
├── .gitignore                     # Git exclusion rules
├── LICENSE                        # MIT Open-Source License
│
├── configs/
│   └── default.yaml               # Centralized hyperparameters & paths
│
├── models/
│   ├── __init__.py                # Package module exports
│   ├── attention_unet.py          # PyTorch Attention U-Net implementation
│   └── cbam.py                    # Convolutional Block Attention Module (CBAM)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Band normalization & NDSI/NDVI calculations
│   ├── dataset.py                 # PyTorch Dataset loader for .npy tensors
│   ├── losses.py                  # Hybrid BCE, Soft Dice & Focal Tversky Loss
│   ├── metrics.py                 # Dice, IoU, Precision, Recall computation
│   ├── engine.py                  # Training & validation epoch iteration loops
│   └── utils.py                   # Reproducibility seeders & YAML parser
│
├── scripts/
│   ├── inspect_geotiff.py         # Geospatial raster inspection tool
│   ├── prepare_dataset.py         # Synthetic patch generator for smoke testing
│   ├── acquire_satellite_data.py  # STAC multi-satellite data acquisition pipeline
│   ├── train.py                   # Main training script with checkpoint saving
│   ├── evaluate.py                # Metric evaluation & threshold analysis
│   ├── predict.py                 # Full-scene sliding-window inference & export
│   └── statistical_sensor_analysis.py # Multi-satellite statistical analysis suite
│
├── tests/
│   ├── test_dataset.py            # Unit test for data loader
│   ├── test_losses.py             # Unit test for loss functions
│   ├── test_metrics.py            # Unit test for mathematical metrics
│   └── test_model.py              # Unit test for model forward pass
│
├── notebooks/
│   └── glacier_segmentation_demo.ipynb # Interactive Google Colab GPU notebook
│
├── data/
│   └── README.md                  # Data placement instructions
├── outputs/
│   ├── checkpoints/               # Saved model weight checkpoints (.pth)
│   ├── predictions/               # Predicted masks and visual overlays (.png, .npy)
│   └── statistical_analysis/      # Multi-satellite comparison metrics, tests & figures
│       ├── figures/               # Publication-grade comparison plots
│       ├── sensor_comparison_metrics.csv
│       ├── patch_level_metrics.csv
│       ├── descriptive_statistics.csv
│       ├── statistical_tests.csv
│       ├── confidence_intervals.csv
│       └── statistical_analysis_report.md
└── docs/
    ├── ARCHITECTURE.md            # Detailed architectural breakdown
    ├── DATASET_GUIDE.md           # Multispectral band specification guide
    └── TRAINING_GUIDE.md          # Training instructions
```

---

## ⚙️ 10. Installation & Local Setup

```bash
# 1. Clone repository
git clone https://github.com/rameshwar17403/Attention-U-Net-Glacier-Segmentation.git
cd Attention-U-Net-Glacier-Segmentation

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# .\venv\Scripts\Activate.ps1  # On Windows PowerShell

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 11. Running Tests, Training, Evaluation & Inference

```bash
# Set PYTHONPATH
export PYTHONPATH=$PWD          # On Linux/macOS
# $env:PYTHONPATH = "$PWD"     # On Windows PowerShell

# 1. Run unit test suite (8 tests)
python -m unittest discover tests

# 2. Extract multi-satellite patches
python scripts/acquire_satellite_data.py --output-dir data/satellite_patches --num-patches 30

# 3. Execute multi-satellite statistical analysis
python scripts/statistical_sensor_analysis.py \
    --s2-dir data/satellite_patches/s2_patches \
    --landsat-dir data/satellite_patches/landsat_patches \
    --mask-dir data/satellite_patches/masks \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output-dir outputs/statistical_analysis \
    --config configs/default.yaml \
    --threshold 0.70 \
    --bootstrap-iters 10000

# 4. Train the Attention U-Net
python scripts/train.py --config configs/default.yaml

# 5. Evaluate the saved checkpoint
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth

# 6. Run full-scene inference & visualization
python scripts/predict.py --checkpoint outputs/checkpoints/best_model.pth
```

---

## ☁️ 12. Google Colab Workflow

Upload `notebooks/glacier_segmentation_demo.ipynb` to Google Colab, select a **T4 GPU** runtime, and execute cells interactively.

---

## 📚 13. Data Attribution & References

- **Sentinel-2 Data**: Copernicus Sentinel-2 Level-2A surface reflectance data accessed via Google Earth Engine / AWS Open Data.
- **Landsat Data**: USGS/NASA Landsat 8/9 OLI Collection 2 Level-2 Surface Reflectance and NASA Harmonized Landsat-Sentinel (HLS v2.0).
- **Reference Labels**: Randolph Glacier Inventory (RGI 7.0/6.0 Region 14: South Asia West) vector polygons provided by GLIMS / NSIDC.
- **Architecture Reference**: O. Oktay et al., _“Attention U-Net: Learning Where to Look for Pancreas Segmentation”_ (arXiv:1804.03999).
- **CBAM Reference**: S. Woo et al., _“CBAM: Convolutional Block Attention Module”_ (ECCV 2018).

---

## 📜 14. License

This project is open-source under the [MIT License](LICENSE).
