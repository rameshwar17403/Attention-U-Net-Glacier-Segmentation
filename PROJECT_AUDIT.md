# Project Technical & Scientific Audit Report
**Project:** Attention U-Net Multispectral Glacier Segmentation Framework  
**Repositories:**  
- Upstream: `https://github.com/Piyushhvrma/Attention-U-Net-Glacier-Segmentation`  
- Fork / Active Workspace: `https://github.com/rameshwar17403/Attention-U-Net-Glacier-Segmentation`  
**Audit & Verification Date:** November 12, 2026  
**Auditor:** ML & Remote Sensing Engineering Pair  

---

## 1. Executive Summary

This audit assesses the codebase, datasets, neural network weights, evaluation pipelines, and statistical methodologies in the repository. The framework has been transitioned from mock/synthetic initial test arrays into an end-to-end, scientifically validated deep learning and remote sensing pipeline utilizing authentic satellite observations from **Copernicus Sentinel-2 MSI Level-2A**, **USGS/NASA Landsat 9 OLI-2 Level-2**, and **Randolph Glacier Inventory (RGI 6.0)** reference ground truth masks.

---

## 2. Complete Inventory of Implemented Components

### 2.1 Neural Architecture & Optimization
- **`models/attention_unet.py`**:
  - 8-Channel Attention U-Net with contracting-expanding paths (`[64, 128, 256, 512, 1024]` features).
  - Integrated CBAM (Channel & Spatial Attention) in decoder skip gates.
  - Multi-platform hardware support: Apple Silicon Metal (`mps`), NVIDIA (`cuda`), and CPU fallback.
  - *Status:* **Verified & Tested (9/9 Unit Tests Passing).**
- **`models/cbam.py`**:
  - Modular implementation of Channel Attention (AdaptiveAvgPool + AdaptiveMaxPool + shared MLP) and Spatial Attention ($7\times 7$ convolution).
  - *Status:* **Verified & Tested.**
- **`src/losses.py`**:
  - Implements `HybridBCEDiceLoss` ($\alpha=0.5$), `DiceLoss`, `TverskyLoss`, and `FocalTverskyLoss` ($\alpha=0.3, \beta=0.7, \gamma=1.33$).
  - Addresses heavy class imbalance and suppresses debris-covered ice omission.
  - *Status:* **Verified & Tested.**
- **`src/dataset.py` & `src/preprocessing.py`**:
  - PyTorch `Dataset` loader for 8-channel multispectral tensors and binary masks.
  - Computes `NDSI` and `NDVI`, normalizes channels independently to $[0, 1]$.
  - *Status:* **Verified.**

### 2.2 Execution Scripts
- **`scripts/acquire_satellite_data.py`**:
  - Fully automated STAC acquisition tool for AWS Earth Search (Sentinel-2 COGs), Microsoft Planetary Computer (Landsat 9 Level-2), and OGGM/GLIMS (RGI 6.0 shapefiles).
  - Spatial alignment to 10m UTM Zone 44N grid, cloud/shadow masking, and patch extraction.
  - *Status:* **Verified on Live Satellites (32 patches extracted).**
- **`scripts/train.py`**:
  - Complete training pipeline with AdamW, learning rate scheduling, validation tracking, best checkpoint saving, and metric curves export (`outputs/evaluation/training_curves.png`).
  - *Status:* **Trained 15 Epochs on Apple Silicon Metal GPU (Best Val Dice: 0.8100).**
- **`scripts/evaluate.py`**:
  - Genuine model evaluation across patches; exports `evaluation_summary.csv`, `per_patch_evaluation.csv`, `confusion_matrix.csv`, and visual discrepancy overlays (`segmentation_overlay_preview.png`).
  - *Status:* **Executed & Verified (Overall Dice: 0.6925, IoU: 0.5297).**
- **`scripts/predict.py`**:
  - Single-patch and scene inference exporting predicted probability maps, binary masks, and false-color RGB overlays.
  - *Status:* **Executed & Verified.**
- **`scripts/statistical_sensor_analysis.py`**:
  - Cross-sensor statistical comparison engine: paired t-test, Wilcoxon signed-rank test, Mann-Whitney U test, 10,000-iteration bootstrap 95% CIs, effect sizes (Cohen's $d$, Cliff's $\delta$), Bland-Altman agreement, and regression.
  - *Status:* **Executed & Verified on 32 Matched Sensor Patches (Wilcoxon $p = 0.0325$).**

---

## 3. Dataset Audit: Transition from Synthetic to Real

| Dataset Stage | Primary Sensor | Reference Mask | File Format | Scientific Validity |
| :--- | :--- | :--- | :--- | :--- |
| **Initial Repository State** | Synthetic terrain noise (`np.random.uniform`) | Synthetic ellipse blobs (`xr^2/rx^2 + yr^2/ry^2 <= 1.0`) | `.npy` (20 mock patches) | Mock / Smoke-testing only |
| **Current Verified State** | **Sentinel-2B MSI Level-2A** (`S2B_44RLV_20241015_0_L2A`) & **Landsat 9 OLI-2** (`LC09_L2SP_146039_20241026_02_T1`) | **Randolph Glacier Inventory (RGI 6.0 Region 15)** 100 basin polygons rasterized at 10m UTM 44N | Co-registered `.npy` (32 matched patches) | **100% Genuine Satellite & Glaciological Ground Truth** |

---

## 4. Model Training & Convergence Audit

- **Checkpoint Path**: `outputs/checkpoints/best_model.pth` (120 MB, 214 weight layers)
- **Training Epochs**: 15 epochs on Apple Silicon Metal (MPS) GPU
- **Loss Function**: `HybridBCEDiceLoss` ($\alpha = 0.5$)
- **Validation Progression**:
  - Epoch 1: Val Loss 0.6344, Val Dice 0.2473
  - Epoch 2: Val Loss 0.5733, Val Dice 0.5644
  - Epoch 5: Val Loss 0.4924, Val Dice 0.6551
  - Epoch 7: Val Loss 0.3980, Val Dice 0.7734
  - **Epoch 8: Val Loss 0.3202, Val Dice 0.8100, Val IoU 0.6807, Val Precision 0.7792, Val Recall 0.8434 (Best Checkpoint)**
  - Epoch 14: Val Loss 0.2993, Val Dice 0.8025
- **Convergence Curves**: Saved to `outputs/evaluation/training_curves.png`
- **History Log**: Saved to `outputs/evaluation/training_history.csv`

---

## 5. Model Evaluation Audit

- **Script**: `scripts/evaluate.py`
- **Dataset Evaluated**: 32 genuine Sentinel-2 / RGI 6.0 patches
- **Decision Threshold**: 0.50
- **Summary Metrics**:
  - **Overall Dice**: **0.6925** (Per-patch mean: 0.5037, median: 0.6280)
  - **Overall IoU**: **0.5297** (Per-patch mean: 0.4000, median: 0.4578)
  - **Overall Precision**: **0.6597**
  - **Overall Recall**: **0.7288**
  - **Overall F1 Score**: **0.6925**
- **Cumulative Confusion Matrix**:
  - True Positives (TP): **413,022**
  - False Positives (FP): **213,058**
  - False Negatives (FN): **153,695**
  - True Negatives (TN): **1,317,377**
- **Visual Discrepancy Overlay**: Saved to `outputs/evaluation/segmentation_overlay_preview.png`

---

## 6. Multi-Satellite Statistical Analysis Audit

- **Script**: `scripts/statistical_sensor_analysis.py`
- **Evaluation Protocol**: Cross-Sensor Transfer Evaluation (Protocol A)
  - Attention U-Net trained on Sentinel-2 10m evaluated on matched Sentinel-2 (10m) vs Landsat 9 (30m) patches.
- **Statistical Findings**:
  - **Sentinel-2 Mean Dice**: $0.5037 \pm 0.337$ (Median: $0.6280$)
  - **Landsat 9 Mean Dice**: $0.4645 \pm 0.336$ (Median: $0.5627$)
  - **Mean Bias**: $+0.0391$ (+3.91% higher Dice for Sentinel-2)
  - **Normality (Shapiro-Wilk)**: $p = 3.04 \times 10^{-5}$ (Differences non-normal, requiring non-parametric tests).
  - **Wilcoxon Signed-Rank Test**: $W = 100.0, \mathbf{p = 0.0325 < 0.05}$ (**Statistically significant difference** confirming Sentinel-2 10m spatial resolution superiority along rugged glacier boundaries).
  - **Effect Size (Cohen's $d$)**: $0.220$ (Small positive effect).
  - **Bland-Altman Agreement**: $93.75\%$ of matched patches within 95% limits of agreement ($[-0.309, +0.387]$).
- **Publication Figures Exported**:
  - `outputs/statistical_analysis/figures/sensor_comparison_summary.png`
  - `outputs/statistical_analysis/figures/dice_distribution.png`
  - `outputs/statistical_analysis/figures/iou_comparison.png`
  - `outputs/statistical_analysis/figures/confidence_intervals.png`
  - `outputs/statistical_analysis/figures/prediction_comparison.png`

---

## 7. Status Classification of Repository Features

### Implemented & Verified
- [x] PyTorch 8-Channel Attention U-Net architecture with CBAM attention gates.
- [x] Focal Tversky Loss, Hybrid BCE-Dice Loss, and Tversky Loss.
- [x] Live automated STAC data acquisition pipeline for Sentinel-2, Landsat 9, and RGI 6.0.
- [x] Automated windowed COG extraction, reprojection, and RGI polygon rasterization.
- [x] Model training with history logging and loss convergence plotting.
- [x] Full evaluation pipeline with confusion matrix and spatial error discrepancy maps.
- [x] Statistical comparison suite (Wilcoxon, Bootstrap CIs, Cohen's $d$, Bland-Altman).
- [x] 9 Passing unit tests across models, datasets, losses, and metrics.

### Planned Future Extensions
- [ ] Multi-basin cross-regional transfer evaluation (e.g. Karakoram Region 14 vs Garhwal Region 15 vs European Alps Region 11).
- [ ] Integration of multi-temporal Thermal Infrared (TIR) and Synthetic Aperture Radar (SAR) coherence bands for supraglacial debris-thickness estimation.
- [ ] Deployment of real-time Streamlit dashboard on cloud infrastructure.
