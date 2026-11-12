# Multi-Satellite Cryospheric Dataset Storage & Google Drive Integration Guide

**Project:** Attention U-Net Multispectral Glacier Segmentation Framework  
**Case Study:** Gangotri Glacier Catchment, Uttarakhand, India (Garhwal Himalaya)  
**Primary Satellite:** Copernicus Sentinel-2 MSI Level-2A (Surface Reflectance)  
**Secondary Satellite:** USGS/NASA Landsat 8/9 OLI/OLI-2 Level-2 Surface Reflectance  
**Ground Truth Reference:** Randolph Glacier Inventory (RGI 6.0 Region 15: South Asia East, ID: `RGI60-15.06881`)  

---

## 1. Storage Policy: Google Drive & Persistent Storage (Zero Satellite Data on GitHub)

In compliance with open science data management and GitHub file size constraints:
1. **GitHub Repository Scope:** Tracks source code, model definitions, training/evaluation pipelines, statistical analysis scripts, dataset inventories (`dataset_inventory.csv`), metadata logs, Jupyter notebooks, publication figures, and presentations.
2. **Satellite Binary Exclusions:** Raw satellite GeoTIFFs, resampled multi-band GeoTIFFs, 8-channel `.npy` patch arrays, and rasterized ground truth mask arrays are **strictly excluded from Git tracking** via `.gitignore`.
3. **Persistent Cloud Storage:** Complete large-scale dataset archives (exceeding 1,000 valid genuine patches) are preserved in persistent cloud storage (Google Drive, AWS S3, and institutional mirrors).

---

## 2. Standardized Dataset Directory Structure

```text
data/
├── dataset_inventory.csv               # Complete patch-by-patch inventory & spatial catalog
├── dataset_verification_report.json    # Verification report certifying >= 1,000 valid patches
├── README.md                           # Dataset documentation & band layout
├── raw/
│   └── rgi15/                          # Official RGI 6.0 Region 15 shapefile and attributes
├── patches/                            # Large-scale Sentinel-2 multi-temporal patches (>=1,000)
│   ├── s2_2021_p0000_patch.npy         # 8-channel float32 array (256x256)
│   ├── s2_2021_p0000_mask.npy          # Binary RGI ground truth mask (256x256)
│   └── ...
└── satellite_patches/                  # Paired cross-sensor patches for Objective 2
    ├── acquisition_metadata.json       # Sensor acquisition & spatial harmonization parameters
    ├── s2_patches/                     # Harmonized 30m Sentinel-2 patches
    ├── landsat_patches/                # Harmonized 30m Landsat 8/9 patches
    └── masks/                          # Matched RGI 6.0 binary ground truth masks
```

---

## 3. Google Colab Execution & Google Drive Mounting

To train the Attention U-Net model or execute the 8-notebook research suite in Google Colab with free GPU acceleration:

### Step 1: Mount Google Drive
```python
from google.colab import drive
import os

drive.mount('/content/drive')
GDRIVE_DIR = '/content/drive/MyDrive/Attention_UNet_Glacier_Data'
os.makedirs(GDRIVE_DIR, exist_ok=True)
print(f"[+] Google Drive mounted at: {GDRIVE_DIR}")
```

### Step 2: Clone GitHub Codebase
```bash
!git clone https://github.com/rameshwar17403/Attention-U-Net-Glacier-Segmentation.git
%cd Attention-U-Net-Glacier-Segmentation
!pip install -r requirements.txt
```

### Step 3: Link or Sync Persistent Dataset
```python
# Create a symbolic link from Google Drive dataset to local repository
!ln -s /content/drive/MyDrive/Attention_UNet_Glacier_Data/patches data/patches
!ln -s /content/drive/MyDrive/Attention_UNet_Glacier_Data/satellite_patches data/satellite_patches
print("[+] Dataset linked to persistent Google Drive storage.")
```

### Step 4: Run Training or Evaluation
```bash
# Execute training on Apple MPS or NVIDIA CUDA GPU
!python scripts/train.py --epochs 15 --batch-size 8 --lr 0.0002

# Execute independent test set evaluation
!python scripts/evaluate.py --split test --threshold 0.50
```

---

## 4. Automated Dataset Synchronization Script

A helper script `scripts/sync_dataset_to_gdrive.py` is included in the repository:
```bash
python scripts/sync_dataset_to_gdrive.py --target-dir /path/to/GoogleDrive/Attention_UNet_Glacier_Data
```

This utility verifies patch integrity, compresses the patch archive into split-level tarballs (`train_patches.tar.gz`, `val_patches.tar.gz`, `test_patches.tar.gz`), copies metadata, and checks SHA256 checksums to ensure reproducibility.
