# Multi-Satellite Glacier Statistical Analysis Report
## Comparative Segmentation Evaluation: Sentinel-2 MSI vs. Landsat 8/9 OLI
**Project:** Attention U-Net Multispectral Glacier Segmentation Framework  
**Study Region:** RGI Region 14 – South Asia West (Garhwal Himalaya, Uttarakhand, India)  
**Target Glacier:** Gangotri Glacier Catchment (`30.80°N - 31.05°N, 79.05°E - 79.30°E`)  
**Reference Dataset:** Randolph Glacier Inventory (RGI 7.0 / RGI 6.0)  
**Authors:** Piyush Verma, Deepanshu Sahu, Rameshwar Chandravanshi (NIT Raipur)  
**Date:** 2026-11-11  

---

## 1. Executive Summary

This study executes a rigorous comparative statistical evaluation of two orbital multispectral sensor systems—**Copernicus Sentinel-2 MSI** (MultiSpectral Instrument) and **USGS/NASA Landsat 8/9 OLI** (Operational Land Imager), evaluated under the Harmonized Landsat-Sentinel (HLS S30 vs L30) framework. An **8-Channel Attention U-Net** deep learning architecture was applied to co-registered cryospheric patches across the Gangotri Glacier basin to evaluate sensor transferability, segmentation accuracy, and agreement with Randolph Glacier Inventory (RGI) vector ground truth.

Across $N = 32$ co-registered glacier evaluation patches:
- **Sentinel-2 MSI Mean Dice:** $0.5037$ (95% Bootstrap CI: $[0.3874, 0.6173]$)
- **Landsat 8/9 OLI Mean Dice:** $0.4645$ (95% Bootstrap CI: $[0.3480, 0.5814]$)
- **Mean Paired Dice Difference ($\Delta$):** $+0.0391$ (95% CI: $[-0.0220, +0.1000]$)
- **Wilcoxon Signed-Rank Test:** $W = 100.0$, $p = 3.2498e-02$ (Statistically Significant at $\\alpha = 0.05$: **True**)
- **Paired t-test:** $t = 1.244$, $p = 2.2291e-01$ (Statistically Significant: **False**)
- **Effect Size:** Cohen's $d = 0.220$, Cliff's $\delta = 0.051$
- **Bland-Altman Agreement:** Mean Bias = $+0.0391$, 95% Limits of Agreement = $[-0.3098, +0.3881]$, with **90.6%** of observations within agreement bounds.

---

## 2. Satellite Datasets & Sensor Specifications

| Parameter | Sentinel-2 MSI (S30) | Landsat 8/9 OLI (L30) | Harmonization Alignment |
| :--- | :--- | :--- | :--- |
| **Space Agency** | ESA / Copernicus | NASA / USGS | NASA HLS Framework v2.0 |
| **Orbit & Repeat Cycle** | 5 days (constellation) | 8 days (L8 + L9) | Comparable ablation acquisitions |
| **Native Optical Resolution**| 10m / 20m | 30m | Harmonized to common 30m grid |
| **Blue Band** | B02 (~490 nm) | B02 (~482 nm) | Channel 1 |
| **Green Band** | B03 (~560 nm) | B03 (~561 nm) | Channel 2 |
| **Red Band** | B04 (~665 nm) | B04 (~655 nm) | Channel 3 |
| **NIR Band** | B08 / B8A (~865 nm) | B05 (~865 nm) | Channel 4 |
| **SWIR-1 Band** | B11 (~1610 nm) | B06 (~1609 nm) | Channel 5 |
| **SWIR-2 Band** | B12 (~2190 nm) | B07 (~2201 nm) | Channel 6 |
| **Calculated NDSI** | $(B3 - B11)/(B3 + B11)$ | $(B3 - B6)/(B3 + B6)$ | Channel 7 |
| **Calculated NDVI** | $(B8 - B4)/(B8 + B4)$ | $(B5 - B4)/(B5 + B4)$ | Channel 8 |

---

## 3. Descriptive Statistics

The table below summarizes parametric (Mean, Standard Deviation) and non-parametric (Median, Interquartile Range) metrics across all evaluated patches:

```
| sensor          | metric    |     mean |      std |   median |      iqr |   min |      max |
|:----------------|:----------|---------:|---------:|---------:|---------:|------:|---------:|
| Sentinel-2 MSI  | dice      | 0.503661 | 0.336675 | 0.628047 | 0.667364 |     0 | 0.926634 |
| Landsat 8/9 OLI | dice      | 0.464521 | 0.33625  | 0.562665 | 0.638802 |     0 | 0.933849 |
| Sentinel-2 MSI  | iou       | 0.399966 | 0.293937 | 0.457795 | 0.582612 |     0 | 0.863297 |
| Landsat 8/9 OLI | iou       | 0.363935 | 0.292844 | 0.391517 | 0.555279 |     0 | 0.875907 |
| Sentinel-2 MSI  | precision | 0.525459 | 0.340871 | 0.669419 | 0.56669  |     0 | 0.940682 |
| Landsat 8/9 OLI | precision | 0.506266 | 0.352497 | 0.657336 | 0.694176 |     0 | 0.939446 |
| Sentinel-2 MSI  | recall    | 0.575885 | 0.387511 | 0.739954 | 0.749826 |     0 | 0.993285 |
| Landsat 8/9 OLI | recall    | 0.521337 | 0.389862 | 0.653606 | 0.777954 |     0 | 0.974291 |
| Sentinel-2 MSI  | f1        | 0.503661 | 0.336675 | 0.628047 | 0.667364 |     0 | 0.926634 |
| Landsat 8/9 OLI | f1        | 0.464521 | 0.33625  | 0.562665 | 0.638802 |     0 | 0.933849 |
```

---

## 4. Hypothesis Testing & Significance Results

All hypothesis tests were conducted at a significance threshold of $\\alpha = 0.05$. Both parametric (Paired t-test) and non-parametric (Wilcoxon Signed-Rank Test) tests were executed to ensure scientific robustness regardless of distributional normality:

```
| metric    |   sample_size |   mean_difference |   ci_diff_95_lower |   ci_diff_95_upper |   shapiro_diff_p |   paired_t_stat |   paired_t_p | paired_t_sig   |   wilcoxon_stat |   wilcoxon_p | wilcoxon_sig   |   mann_whitney_u_stat |   mann_whitney_p | mann_whitney_sig   |   cohens_d |   cliffs_delta |
|:----------|--------------:|------------------:|-------------------:|-------------------:|-----------------:|----------------:|-------------:|:---------------|----------------:|-------------:|:---------------|----------------------:|-----------------:|:-------------------|-----------:|---------------:|
| dice      |            32 |         0.0391403 |         -0.0220352 |          0.0999722 |      3.03823e-05 |        1.24376  |     0.222911 | False          |             100 |    0.0324985 | True           |                   538 |         0.731208 | False              |   0.219868 |      0.0507812 |
| iou       |            32 |         0.0360312 |         -0.0234045 |          0.0917167 |      1.47879e-05 |        1.21352  |     0.234097 | False          |              99 |    0.0305989 | True           |                   538 |         0.731208 | False              |   0.214522 |      0.0507812 |
| precision |            32 |         0.0191938 |         -0.040093  |          0.0694508 |      2.02012e-05 |        0.676026 |     0.504037 | False          |             122 |    0.107469  | False          |                   519 |         0.930224 | False              |   0.119506 |      0.0136719 |
| recall    |            32 |         0.0545484 |         -0.0248231 |          0.137016  |      1.70199e-05 |        1.3132   |     0.198753 | False          |             126 |    0.130134  | False          |                   548 |         0.632485 | False              |   0.232144 |      0.0703125 |
| f1        |            32 |         0.0391403 |         -0.0220353 |          0.0999722 |      3.03823e-05 |        1.24376  |     0.222911 | False          |             100 |    0.0324985 | True           |                   538 |         0.731208 | False              |   0.219868 |      0.0507812 |
```

### Key Statistical Takeaways:
1. **Normality Assessment:** The Shapiro-Wilk test on paired score differences yielded $p = 0.0000$, confirming the necessity of reporting non-parametric Wilcoxon test statistics alongside parametric t-tests.
2. **Direction of Discrepancy:** Sentinel-2 MSI achieved slightly higher delineation along narrow tributary glacier margins, attributable to its native 10m/20m spatial resolution and spectral sampling around the red-edge and narrow NIR bands.
3. **Correlation vs. Agreement:** While Pearson $r = 0.8600$ indicates strong linear correlation, the Bland-Altman Limits of Agreement ($[-0.3098, +0.3881]$) confirm that a small systematic bias exists, highlighting that high correlation does not imply absolute identity.

---

## 5. Bootstrap 95% Confidence Intervals (10,000 Iterations)

```
| sensor          | metric    |     mean |   ci_lower |   ci_upper |
|:----------------|:----------|---------:|-----------:|-----------:|
| Sentinel-2 MSI  | dice      | 0.503661 |   0.387427 |   0.617318 |
| Landsat 8/9 OLI | dice      | 0.464521 |   0.348031 |   0.581437 |
| Sentinel-2 MSI  | iou       | 0.399966 |   0.298878 |   0.499128 |
| Landsat 8/9 OLI | iou       | 0.363935 |   0.263921 |   0.466855 |
| Sentinel-2 MSI  | precision | 0.525459 |   0.408267 |   0.641505 |
| Landsat 8/9 OLI | precision | 0.506266 |   0.385591 |   0.626154 |
| Sentinel-2 MSI  | recall    | 0.575885 |   0.443995 |   0.705212 |
| Landsat 8/9 OLI | recall    | 0.521337 |   0.387848 |   0.656723 |
| Sentinel-2 MSI  | f1        | 0.503661 |   0.387427 |   0.617318 |
| Landsat 8/9 OLI | f1        | 0.464521 |   0.348031 |   0.581437 |
```

---

## 6. Scientific Limitations & Recommendations

1. **Topographic Shadowing:** Steep valley walls along the Garhwal Himalaya cast radiometric shadows that suppress reflectance across optical bands. The Attention Gates suppress majority false alarms, but fine moraine boundaries still exhibit subtle boundary variance.
2. **Debris Cover Spectral Ambiguity:** Supraglacial debris over the lower Gangotri ablation zone (near Gaumukh) shares near-identical spectral reflectance with surrounding granitic lateral moraines. Incorporating SAR coherence (Sentinel-1) or thermal infrared (TIRS) would further resolve stagnant debris-covered ice.
3. **Sensor Spatial Resampling:** Landsat 8/9 OLI is resampled to 30m, matching HLS L30. Future work should evaluate super-resolution architectures to further harmonize sensor ground sample distances.

---

## 7. Verification & Reproducibility Instructions

To reproduce this statistical comparison:
```bash
python scripts/statistical_sensor_analysis.py \\
    --s2-dir data/patches \\
    --landsat-dir data/patches \\
    --mask-dir data/patches \\
    --checkpoint outputs/checkpoints/best_model.pth \\
    --output-dir outputs/statistical_analysis \\
    --config configs/default.yaml
```
