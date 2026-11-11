# Dataset Guide: Harmonized Multispectral Satellite Data & RGI Ground Truth

## Overview
This project uses **8-channel multispectral patches** ($256 \times 256$ pixels, 30 m ground resolution) derived from Copernicus Sentinel-2 MSI Level-2A and Landsat 8/9 OLI Level-2 surface reflectance observations, paired with Randolph Glacier Inventory (RGI 6.0) vector outlines.

### Geographic & Glaciological Specification: RGI Region 15
- **Study Area:** Gangotri Glacier Basin, Garhwal Himalaya, Uttarakhand, India (`[79.05°E, 30.80°N, 79.30°E, 31.05°N]`).
- **Inventory Region:** Although early working drafts colloquially cited Region 14 (South Asia West), Gangotri Glacier (snout at 30.93°N, 79.08°E, stretching southeastward to Chaukhamba peaks) is officially cataloged in **RGI 6.0 Region 15: South Asia East** under ID **`RGI60-15.06881`** (area: 121.89 km²). Region 14 terminates westward in Himachal Pradesh and contains zero glaciers in the Gangotri basin. Ground truth outlines were extracted from the official OGGM/GLIMS Region 15 inventory.

### NASA HLS Common 30m Spatial Harmonization Grid
- **Coordinate Reference System:** UTM Zone 44N (`EPSG:32644`)
- **Pixel Grid:** Snapped to integer multiples of 30.0 meters:
  - Extent: X $[313410.0, 321090.0]$, Y $[3408660.0, 3436800.0]$
  - Dimensions: $938 \times 813$ pixels
- **Sentinel-2 Resampling:** 10m bands (`B02`, `B03`, `B04`, `B08`) and 20m bands (`B11`, `B12`) are resampled via bilinear interpolation to the exact 30m grid.
- **Landsat 9 Alignment:** Native 30m bands (`SR_B2` through `SR_B7`) are projected and aligned to the exact identical pixel grid.

### 8-Channel Multispectral Tensor Specification
1. `B2` (Blue) - ~490 nm (S2) / ~482 nm (L9)
2. `B3` (Green) - ~560 nm (S2) / ~561 nm (L9)
3. `B4` (Red) - ~665 nm (S2) / ~655 nm (L9)
4. `B8` (NIR) - ~842 nm (S2 B8) / ~865 nm (L9 B5)
5. `B11` (SWIR1) - ~1610 nm (S2) / ~1609 nm (L9 B6)
6. `B12` (SWIR2) - ~2190 nm (S2) / ~2201 nm (L9 B7)
7. `NDSI` - Normalized Difference Snow Index (`(B3 - B11) / (B3 + B11)`)
8. `NDVI` - Normalized Difference Vegetation Index (`(B8 - B4) / (B8 + B4)`)

