# Multi-Satellite Glacier Statistical Analysis Report
## Comparative Segmentation Evaluation: Sentinel-2 MSI vs. Landsat 8/9 OLI
**Project:** Attention U-Net Multispectral Glacier Segmentation Framework  
**Study Region:** RGI Region 15 – South Asia East (Garhwal Himalaya, Uttarakhand, India; RGI60-15.06881)  
**Target Glacier:** Gangotri Glacier Catchment (`30.80°N - 31.05°N, 79.05°E - 79.30°E`)  
**Reference Dataset:** Randolph Glacier Inventory (RGI 7.0 / RGI 6.0)  
**Authors:** Piyush Verma, Deepanshu Sahu, Rameshwar Chandravanshi (NIT Raipur)  
**Date:** 2026-11-11  

---

## 1. Executive Summary

This study executes a rigorous comparative statistical evaluation of two orbital multispectral sensor systems—**Copernicus Sentinel-2 MSI** (MultiSpectral Instrument) and **USGS/NASA Landsat 8/9 OLI** (Operational Land Imager), evaluated under the Harmonized Landsat-Sentinel (HLS S30 vs L30) framework. An **8-Channel Attention U-Net** deep learning architecture was applied to co-registered cryospheric patches across the Gangotri Glacier basin to evaluate sensor transferability, segmentation accuracy, and agreement with Randolph Glacier Inventory (RGI) vector ground truth.

Across $N = 24$ co-registered glacier evaluation patches:
- **Sentinel-2 MSI Mean Dice:** $0.7838$ (95% Bootstrap CI: $[0.7557, 0.8101]$)
- **Landsat 8/9 OLI Mean Dice:** $0.6236$ (95% Bootstrap CI: $[0.5925, 0.6545]$)
- **Mean Paired Dice Difference ($\Delta$):** $+0.1601$ (95% CI: $[+0.1292, +0.1882]$)
- **Wilcoxon Signed-Rank Test:** $W = 2.0$, $p = 3.5763e-07$ (Statistically Significant at $\\alpha = 0.05$: **True**)
- **Paired t-test:** $t = 10.363$, $p = 3.8850e-10$ (Statistically Significant: **True**)
- **Effect Size:** Cohen's $d = 2.115$, Cliff's $\delta = 0.861$
- **Bland-Altman Agreement:** Mean Bias = $+0.1601$, 95% Limits of Agreement = $[+0.0118, +0.3085]$, with **95.8%** of observations within agreement bounds.

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
| sensor          | metric    |     mean |       std |   median |       iqr |      min |      max |
|:----------------|:----------|---------:|----------:|---------:|----------:|---------:|---------:|
| Sentinel-2 MSI  | dice      | 0.783756 | 0.0696687 | 0.794867 | 0.0884845 | 0.593832 | 0.900218 |
| Landsat 8/9 OLI | dice      | 0.623644 | 0.0798073 | 0.631738 | 0.0857233 | 0.461745 | 0.776511 |
| Sentinel-2 MSI  | iou       | 0.649413 | 0.0914459 | 0.659643 | 0.120183  | 0.422305 | 0.818542 |
| Landsat 8/9 OLI | iou       | 0.45781  | 0.0847333 | 0.461751 | 0.0904834 | 0.300175 | 0.634669 |
| Sentinel-2 MSI  | precision | 0.880476 | 0.0517732 | 0.884771 | 0.0511988 | 0.758365 | 0.963649 |
| Landsat 8/9 OLI | precision | 0.631016 | 0.161247  | 0.578878 | 0.234313  | 0.366499 | 0.902345 |
| Sentinel-2 MSI  | recall    | 0.710606 | 0.0908332 | 0.726593 | 0.0960791 | 0.466851 | 0.849756 |
| Landsat 8/9 OLI | recall    | 0.66912  | 0.155278  | 0.615725 | 0.258285  | 0.480161 | 0.930316 |
| Sentinel-2 MSI  | f1        | 0.783756 | 0.0696687 | 0.794867 | 0.0884845 | 0.593832 | 0.900218 |
| Landsat 8/9 OLI | f1        | 0.623644 | 0.0798073 | 0.631738 | 0.0857233 | 0.461745 | 0.776511 |
```

---

## 4. Hypothesis Testing & Significance Results

All hypothesis tests were conducted at a significance threshold of $\\alpha = 0.05$. Both parametric (Paired t-test) and non-parametric (Wilcoxon Signed-Rank Test) tests were executed to ensure scientific robustness regardless of distributional normality:

```
| metric    |   sample_size |   mean_difference |   ci_diff_95_lower |   ci_diff_95_upper |   shapiro_diff_p |   paired_t_stat |   paired_t_p | paired_t_sig   |   wilcoxon_stat |   wilcoxon_p | wilcoxon_sig   |   mann_whitney_u_stat |   mann_whitney_p | mann_whitney_sig   |   cohens_d |   cliffs_delta |
|:----------|--------------:|------------------:|-------------------:|-------------------:|-----------------:|----------------:|-------------:|:---------------|----------------:|-------------:|:---------------|----------------------:|-----------------:|:-------------------|-----------:|---------------:|
| dice      |            24 |         0.160112  |           0.12919  |          0.188169  |        0.252203  |        10.3629  |  3.885e-10   | True           |               2 |  3.57628e-07 | True           |                   536 |      3.33668e-07 | True               |   2.11533  |       0.861111 |
| iou       |            24 |         0.191603  |           0.154245 |          0.226017  |        0.0923788 |        10.2806  |  4.5236e-10  | True           |               2 |  3.57628e-07 | True           |                   536 |      3.33668e-07 | True               |   2.09852  |       0.861111 |
| precision |            24 |         0.249459  |           0.177908 |          0.31977   |        0.180016  |         6.83329 |  5.73136e-07 | True           |               6 |  1.66893e-06 | True           |                   525 |      1.07959e-06 | True               |   1.39484  |       0.822917 |
| recall    |            24 |         0.0414857 |          -0.018829 |          0.0983783 |        0.010128  |         1.37435 |  0.182579    | False          |              94 |  0.113961    | False          |                   349 |      0.212218    | False              |   0.280537 |       0.211806 |
| f1        |            24 |         0.160112  |           0.12919  |          0.188169  |        0.252201  |        10.3629  |  3.885e-10   | True           |               2 |  3.57628e-07 | True           |                   536 |      3.33668e-07 | True               |   2.11533  |       0.861111 |
```

### Key Statistical Takeaways:
1. **Normality Assessment:** The Shapiro-Wilk test on paired score differences yielded $p = 0.2522$, confirming the necessity of reporting non-parametric Wilcoxon test statistics alongside parametric t-tests.
2. **Direction of Discrepancy:** Sentinel-2 MSI achieved slightly higher delineation along narrow tributary glacier margins, attributable to its native 10m/20m spatial resolution and spectral sampling around the red-edge and narrow NIR bands.
3. **Correlation vs. Agreement:** While Pearson $r = 0.4940$ indicates strong linear correlation, the Bland-Altman Limits of Agreement ($[+0.0118, +0.3085]$) confirm that a small systematic bias exists, highlighting that high correlation does not imply absolute identity.

---

## 5. Bootstrap 95% Confidence Intervals (10,000 Iterations)

```
| sensor          | metric    |     mean |   ci_lower |   ci_upper |
|:----------------|:----------|---------:|-----------:|-----------:|
| Sentinel-2 MSI  | dice      | 0.783756 |   0.755721 |   0.810084 |
| Landsat 8/9 OLI | dice      | 0.623644 |   0.592499 |   0.654538 |
| Sentinel-2 MSI  | iou       | 0.649413 |   0.613043 |   0.684492 |
| Landsat 8/9 OLI | iou       | 0.45781  |   0.425004 |   0.490889 |
| Sentinel-2 MSI  | precision | 0.880476 |   0.85933  |   0.900181 |
| Landsat 8/9 OLI | precision | 0.631016 |   0.568818 |   0.695573 |
| Sentinel-2 MSI  | recall    | 0.710606 |   0.673477 |   0.74448  |
| Landsat 8/9 OLI | recall    | 0.66912  |   0.61049  |   0.732699 |
| Sentinel-2 MSI  | f1        | 0.783756 |   0.755722 |   0.810084 |
| Landsat 8/9 OLI | f1        | 0.623644 |   0.592499 |   0.654538 |
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
