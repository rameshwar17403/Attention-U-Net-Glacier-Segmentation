"""
Satellite Spatial Harmonization Module (NASA HLS Style)
Harmonizes Sentinel-2 and Landsat 8/9 scenes to a common 30m UTM Zone 44N (EPSG:32644) grid.
Snaps bounding box to integer multiples of 30.0m.
"""

import os
import sys
import numpy as np
from rasterio.warp import transform_bounds
from affine import Affine

# Ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def compute_snapped_grid(bbox_wgs84: list, resolution: float = 30.0, target_crs: str = "EPSG:32644"):
    """
    Computes snapped grid dimensions and affine transform for target bounding box.
    bbox_wgs84: [min_lon, min_lat, max_lon, max_lat]
    """
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", target_crs, *bbox_wgs84)
    snap_xmin = float(np.floor(xmin / resolution) * resolution)
    snap_ymin = float(np.floor(ymin / resolution) * resolution)
    snap_xmax = float(np.ceil(xmax / resolution) * resolution)
    snap_ymax = float(np.ceil(ymax / resolution) * resolution)

    width = int(round((snap_xmax - snap_xmin) / resolution))
    height = int(round((snap_ymax - snap_ymin) / resolution))
    transform = Affine.translation(snap_xmin, snap_ymax) * Affine.scale(resolution, -resolution)

    return {
        "target_crs": target_crs,
        "resolution": resolution,
        "width": width,
        "height": height,
        "bounds": [snap_xmin, snap_ymin, snap_xmax, snap_ymax],
        "transform": transform
    }


def main():
    bbox = [78.95, 30.70, 79.45, 31.15]
    grid = compute_snapped_grid(bbox, resolution=30.0)
    print("=" * 60)
    print("🛰️ HARMONIZED NASA HLS 30M GRID GEOMETRY (GANGOTRI BASIN)")
    print(f"CRS:        {grid['target_crs']}")
    print(f"Resolution: {grid['resolution']} meters")
    print(f"Grid Size:  {grid['height']} rows x {grid['width']} cols ({grid['height']*grid['width']:,} pixels)")
    print(f"UTM Bounds: {grid['bounds']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
