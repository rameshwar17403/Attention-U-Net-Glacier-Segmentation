# Dataset Directory & Preprocessing Guide

This directory manages training, validation, and test patches for the **Attention U-Net Glacier Segmentation** framework.

## Genuine Satellite & Reference Data Sources

1. **Sentinel-2 MSI Level-2A (Primary Sensor)**
   - Source: Copernicus / AWS Earth Search Cloud-Optimized GeoTIFFs (`sentinel-cogs`)
   - Scene: `S2B_44RLV_20241015_0_L2A` (Acquired Oct 15, 2024; Cloud Cover: 0.02%)
   - Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2), SCL
   - Harmonization: Resampled to common 30m UTM Zone 44N grid via bilinear interpolation.

2. **Landsat 9 OLI-2 Level-2 (Cross-Sensor Comparison)**
   - Source: USGS / Microsoft Planetary Computer STAC (SAS Signed HTTPS)
   - Scene: `LC09_L2SP_146039_20241026_02_T1` (Acquired Oct 26, 2024; Cloud Cover: 1.96%)
   - Bands: SR_B2 (Blue), SR_B3 (Green), SR_B4 (Red), SR_B5 (NIR), SR_B6 (SWIR1), SR_B7 (SWIR2)
   - Harmonization: Native 30m resolution aligned to the identical UTM Zone 44N pixel grid (temporal delta: 11 days).

3. **Ground Truth Glacier Reference Outlines**
   - Source: Randolph Glacier Inventory (RGI 6.0 Region 15: South Asia East) via OGGM / GLIMS mirror
   - Cataloged Glacier ID: `RGI60-15.06881` (Gangotri Glacier, 121.89 km²) plus 99 surrounding intersecting tributary glaciers.
   - Polygons are reprojected from EPSG:4326 to UTM Zone 44N (EPSG:32644) and rasterized at exact 30m resolution.

## Automated Acquisition

To acquire genuine satellite imagery and extract co-registered 8-channel patches for Gangotri Glacier Basin (`[79.05°E, 30.80°N, 79.30°E, 31.05°N]`):

```bash
python scripts/acquire_satellite_data.py \
    --bbox 79.05 30.80 79.30 31.05 \
    --date-range 2024-09-01T00:00:00Z/2024-10-31T23:59:59Z \
    --output-dir data/satellite_patches \
    --patch-size 256
```

## Patch Tensor Format

Each extracted `.npy` patch is an 8-channel float32 tensor of shape `(8, 256, 256)`:
- Channel 0: Blue (B2)
- Channel 1: Green (B3)
- Channel 2: Red (B4)
- Channel 3: Near-Infrared (NIR / B8)
- Channel 4: Shortwave-Infrared 1 (SWIR1 / B11)
- Channel 5: Shortwave-Infrared 2 (SWIR2 / B12)
- Channel 6: Normalized Difference Snow Index ($\text{NDSI} = \frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1}}$)
- Channel 7: Normalized Difference Vegetation Index ($\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$)

Each corresponding `*_mask.npy` is a binary float32 mask of shape `(256, 256)` where `1.0 = Glacier/Snow`, `0.0 = Background/Debris/Rock`.
