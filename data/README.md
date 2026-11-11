# Dataset Directory & Preprocessing Guide

This directory manages training, validation, and test patches for the **Attention U-Net Glacier Segmentation** framework.

## Genuine Satellite & Reference Data Sources

1. **Sentinel-2 MSI Level-2A (Primary Sensor)**
   - Source: Copernicus / AWS Earth Search Cloud-Optimized GeoTIFFs (`sentinel-cogs`)
   - Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2), SCL (Scene Classification Layer)
   - Native Resolution: 10m (B2-B8), 20m (B11-B12) resampled to 10m grid.

2. **Landsat 8/9 OLI Level-2 (Cross-Sensor Comparison)**
   - Source: USGS / Microsoft Planetary Computer STAC
   - Bands: B2 (Blue), B3 (Green), B4 (Red), B5 (NIR), B6 (SWIR1), B7 (SWIR2), QA_Pixel
   - Native Resolution: 30m co-registered to the UTM 44N grid.

3. **Ground Truth Glacier Reference Outlines**
   - Source: Randolph Glacier Inventory (RGI 6.0 Region 15 / Region 14) via OGGM / GLIMS
   - Polygons are reprojected from EPSG:4326 to UTM Zone 44N (EPSG:32644) and rasterized at 10m resolution.

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
