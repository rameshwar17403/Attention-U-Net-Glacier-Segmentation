"""
Automated Multi-Satellite Data Acquisition & Spatial Harmonization Pipeline
Harmonized Landsat-Sentinel (HLS) Common 30m Grid Architecture
Sentinel-2 MSI Level-2A & Landsat 8/9 Level-2 Surface Reflectance
With Randolph Glacier Inventory (RGI 6.0 Region 15) Ground Truth

Author: Remote Sensing & ML Engineering Pair
Project: Attention U-Net Multispectral Glacier Segmentation Framework
Study Area: Gangotri Glacier Basin, Uttarakhand, India (Garhwal Himalaya)

Geographic & Glaciological Note:
Although colloquially referenced in initial drafts under Region 14, Gangotri Glacier
(snout at 30.93°N, 79.08°E, stretching to Chaukhamba peaks) is officially cataloged
under Randolph Glacier Inventory (RGI 6.0) Region 15: South Asia East (ID: RGI60-15.06881).
Region 14 (South Asia West) terminates westward in Himachal Pradesh/Indus Basin.
This script accurately uses Region 15 outlines for ground truth rasterization.

Spatial Harmonization Standards (NASA HLS Style):
- Coordinate Reference System: UTM Zone 44N (EPSG:32644)
- Common Pixel Resolution: Exactly 30.0 meters x 30.0 meters
- Pixel Alignment: Bounds snapped to integer multiples of 30.0m
- Paired Geographic Extent: S2, Landsat, and RGI masks share identical affine transform
- Resampling: Bilinear for continuous reflectance bands; Nearest for categorical masks
- Zero Synthetic Fallback: If genuine Landsat data cannot be acquired, execution terminates.
"""

import os
import sys
import argparse
import json
import zipfile
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform_geom
from rasterio.enums import Resampling
from affine import Affine
import rasterio.features
import shapefile
from shapely.geometry import shape, box

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.preprocessing import compute_spectral_indices


def query_sentinel2_stac(bbox: list, date_range: str, max_cloud: float = 20.0) -> list:
    """Queries AWS Earth Search STAC for Sentinel-2 Level-2A COG scenes."""
    url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "bbox": bbox,
        "datetime": date_range,
        "collections": ["sentinel-2-l2a"],
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 10,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("features", [])
        else:
            print(f"[!] STAC Query HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[!] STAC Query Network Error: {e}")
    return []


def query_landsat_stac(bbox: list, date_range: str, max_cloud: float = 20.0) -> list:
    """Queries Microsoft Planetary Computer STAC for Landsat 8/9 Level-2 scenes."""
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "bbox": bbox,
        "datetime": date_range,
        "collections": ["landsat-c2-l2"],
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 5,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("features", [])
        else:
            print(f"[!] Planetary Computer Query HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[!] Planetary Computer Network Error: {e}")
    return []


def sign_planetary_computer_url(asset_href: str) -> str:
    """Obtains a public Read SAS token from Planetary Computer for a Landsat COG asset."""
    sign_url = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={asset_href}"
    resp = requests.get(sign_url, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("href", asset_href)
    return asset_href


def get_rgi_region15_shapefile(raw_dir: str) -> str:
    """
    Downloads and extracts RGI 6.0 Region 15 (South Asia East, covering Garhwal Himalaya / Gangotri)
    from the official OGGM/GLIMS mirror if not already cached.
    """
    os.makedirs(raw_dir, exist_ok=True)
    shp_candidates = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".shp") and not f.startswith(".")]
    if shp_candidates:
        return shp_candidates[0]

    zip_path = os.path.join(raw_dir, "15_rgi60_SouthAsiaEast.zip")
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 10000000:
        rgi_url = "https://cluster.klima.uni-bremen.de/~oggm/rgi/www.glims.org/RGI/rgi60_files/15_rgi60_SouthAsiaEast.zip"
        print(f"[*] Downloading RGI 6.0 Region 15 (~22 MB) from {rgi_url}...")
        r = requests.get(rgi_url, stream=True, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        print("[+] Download complete.")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)

    shp_candidates = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".shp") and not f.startswith(".")]
    return shp_candidates[0]


def rasterize_rgi_mask(shp_path: str, bbox_wgs84: list, target_shape: tuple, target_transform: Affine, target_crs: str) -> tuple:
    """Rasterizes RGI Region 15 glacier outline polygons onto the harmonized 30m grid."""
    print(f"[*] Loading RGI outlines from {os.path.basename(shp_path)}...")
    sf = shapefile.Reader(shp_path)
    study_box = box(*bbox_wgs84)

    matched_geoms = []
    glacier_details = []
    for sr in sf.iterShapeRecords():
        geom = shape(sr.shape.__geo_interface__)
        if geom.intersects(study_box):
            utm_geom = transform_geom("EPSG:4326", target_crs, sr.shape.__geo_interface__)
            matched_geoms.append((utm_geom, 1))
            name = sr.record.as_dict().get("Name", "")
            rgi_id = sr.record.as_dict().get("RGIId", "")
            area_km2 = sr.record.as_dict().get("Area", 0.0)
            glacier_details.append({"rgi_id": rgi_id, "name": name, "area_km2": float(area_km2)})

    print(f"[+] Matched {len(matched_geoms)} RGI glacier polygons in Gangotri study basin.")
    # Highlight Gangotri Glacier
    gangotri_poly = [g for g in glacier_details if "Gangotri" in g["name"] or g["rgi_id"] == "RGI60-15.06881"]
    if gangotri_poly:
        print(f"    -> Confirmed Primary Glacier: {gangotri_poly[0]['rgi_id']} - {gangotri_poly[0]['name']} ({gangotri_poly[0]['area_km2']:.2f} km²)")

    if not matched_geoms:
        return np.zeros(target_shape, dtype=np.uint8), glacier_details

    mask = rasterio.features.rasterize(
        matched_geoms,
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    )
    return mask, glacier_details


def acquire_harmonized_dataset(
    bbox: list,
    date_range: str,
    output_dir: str,
    patch_size: int = 256,
    stride: int = 64,
    max_cloud: float = 20.0
) -> dict:
    """
    Acquires authentic Sentinel-2 Level-2A and Landsat 8/9 Level-2 observations,
    harmonizes them onto an identical 30.0m grid in UTM Zone 44N (EPSG:32644),
    rasterizes RGI 6.0 Region 15 ground truth masks, and extracts paired patches.

    CRITICAL RULE:
    Zero synthetic fallback. If real Landsat data cannot be acquired, execution halts.
    """
    os.makedirs(output_dir, exist_ok=True)
    s2_dir = os.path.join(output_dir, "s2_patches")
    l8_dir = os.path.join(output_dir, "landsat_patches")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(l8_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    target_crs = "EPSG:32644"
    resolution = 30.0  # Scientific HLS common grid resolution

    print("=" * 80)
    print("🛰️ HARMONIZED MULTI-SATELLITE DATA ACQUISITION PIPELINE (HLS 30m GRID)")
    print(f"   Study Area: Gangotri Glacier Basin {bbox}")
    print(f"   Target CRS: {target_crs} (UTM Zone 44N)")
    print(f"   Common Pixel Resolution: {resolution} meters")
    print(f"   Temporal Window: {date_range}")
    print("=" * 80)

    # 1. Establish the Common 30m Grid snapped to multiples of 30.0m
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", target_crs, *bbox)
    snap_xmin = float(np.floor(xmin / resolution) * resolution)
    snap_ymin = float(np.floor(ymin / resolution) * resolution)
    snap_xmax = float(np.ceil(xmax / resolution) * resolution)
    snap_ymax = float(np.ceil(ymax / resolution) * resolution)

    w_grid = int(round((snap_xmax - snap_xmin) / resolution))
    h_grid = int(round((snap_ymax - snap_ymin) / resolution))
    common_transform = Affine(resolution, 0.0, snap_xmin, 0.0, -resolution, snap_ymax)

    print(f"[+] Harmonized 30m Grid Established:")
    print(f"    Origin (UL): ({snap_xmin:.1f}, {snap_ymax:.1f})")
    print(f"    Dimensions:  {w_grid} cols x {h_grid} rows ({w_grid * h_grid:,} pixels)")
    print(f"    BoundingBox: [{snap_xmin:.1f}, {snap_ymin:.1f}, {snap_xmax:.1f}, {snap_ymax:.1f}]")
    print(f"    Affine:      {common_transform}")

    # 2. Query Sentinel-2 MSI Level-2A
    print("\n[1/5] Searching AWS Earth Search for Sentinel-2 MSI Level-2A...")
    s2_features = query_sentinel2_stac(bbox, date_range, max_cloud=max_cloud)
    if not s2_features:
        raise RuntimeError("Sentinel-2 acquisition failed: No scenes found for specified bbox and date range.")

    s2_scene = s2_features[0]
    s2_id = s2_scene["id"]
    s2_cloud = s2_scene["properties"].get("eo:cloud_cover", 0.0)
    s2_datetime = s2_scene["properties"].get("datetime", "")
    s2_assets = s2_scene["assets"]
    print(f"[+] Selected Sentinel-2 Scene: {s2_id}")
    print(f"    Acquisition Time: {s2_datetime} | Cloud Cover: {s2_cloud:.2f}%")

    # 3. Read & Resample Sentinel-2 Bands to 30m Harmonized Grid
    print("    Reading and downsampling Sentinel-2 bands to common 30m grid...")
    s2_band_keys = [("blue", "B02"), ("green", "B03"), ("red", "B04"), ("nir", "B08"), ("swir16", "B11"), ("swir22", "B12")]
    s2_bands_data = []

    for key, name in s2_band_keys:
        url = s2_assets[key]["href"]
        with rasterio.open(url) as src:
            win = from_bounds(snap_xmin, snap_ymin, snap_xmax, snap_ymax, transform=src.transform)
            arr = src.read(1, window=win, out_shape=(h_grid, w_grid), resampling=Resampling.bilinear)
            # Sentinel-2 Level-2A DN to Surface Reflectance: DN / 10000.0
            sr = np.clip(arr.astype(np.float32) / 10000.0, 0.0, 1.0)
            s2_bands_data.append(sr)

    # Read SCL for cloud/shadow masking
    scl_url = s2_assets["scl"]["href"]
    with rasterio.open(scl_url) as src:
        win = from_bounds(snap_xmin, snap_ymin, snap_xmax, snap_ymax, transform=src.transform)
        scl_arr = src.read(1, window=win, out_shape=(h_grid, w_grid), resampling=Resampling.nearest)

    # Valid pixel mask: mask shadows (3), clouds (8, 9), cirrus (10), saturated/nodata
    cloud_mask_s2 = np.isin(scl_arr, [3, 8, 9, 10])
    valid_mask_s2 = (~cloud_mask_s2) & (s2_bands_data[0] > 0.0)

    s2_stack = np.stack(s2_bands_data, axis=0) # (6, H, W)
    s2_8ch = compute_spectral_indices(s2_stack)
    print(f"[+] Sentinel-2 8-Channel Tensor Harmonized: {s2_8ch.shape} (Physical SR + NDSI + NDVI)")

    # 4. Query & Acquire Landsat 8/9 Level-2 Surface Reflectance (STRICT: NO SYNTHETIC FALLBACK)
    print("\n[2/5] Searching Microsoft Planetary Computer for Landsat 8/9 OLI Level-2...")
    l8_features = query_landsat_stac(bbox, date_range, max_cloud=max_cloud)
    if not l8_features:
        raise RuntimeError(
            "[!] Landsat Acquisition Failed: Zero cloud-free Landsat scenes found on Planetary Computer STAC. "
            "Pipeline halted: Synthetic Landsat fallback is strictly prohibited to ensure scientific integrity."
        )

    l8_scene = l8_features[0]
    l8_id = l8_scene["id"]
    l8_cloud = l8_scene["properties"].get("eo:cloud_cover", 0.0)
    l8_datetime = l8_scene["properties"].get("datetime", "")
    print(f"[+] Selected Landsat Scene: {l8_id}")
    print(f"    Acquisition Time: {l8_datetime} | Cloud Cover: {l8_cloud:.2f}%")

    l8_band_keys = ["blue", "green", "red", "nir08", "swir16", "swir22"]
    l8_bands_data = []

    print("    Reading and aligning Landsat bands to common 30m grid...")
    for b_name in l8_band_keys:
        raw_href = l8_scene["assets"][b_name]["href"]
        signed_href = sign_planetary_computer_url(raw_href)
        with rasterio.open(signed_href) as src:
            l8_bounds = transform_bounds(target_crs, src.crs, snap_xmin, snap_ymin, snap_xmax, snap_ymax)
            win = from_bounds(*l8_bounds, transform=src.transform)
            arr = src.read(1, window=win, out_shape=(h_grid, w_grid), resampling=Resampling.bilinear)
            # USGS Landsat Collection 2 Level 2 SR scale factor: DN * 0.0000275 - 0.2
            sr_l8 = np.clip(arr.astype(np.float32) * 0.0000275 - 0.2, 0.0, 1.0)
            l8_bands_data.append(sr_l8)

    l8_stack = np.stack(l8_bands_data, axis=0) # (6, H, W)
    l8_8ch = compute_spectral_indices(l8_stack)
    print(f"[+] Landsat 8-Channel Tensor Harmonized: {l8_8ch.shape} (Physical SR + NDSI + NDVI)")

    # 5. Rasterize Randolph Glacier Inventory (RGI 6.0 Region 15) Outlines
    print("\n[3/5] Aligning and rasterizing RGI 6.0 Region 15 glacier outlines on 30m grid...")
    rgi_raw_dir = os.path.join(output_dir, "..", "raw", "rgi15")
    shp_path = get_rgi_region15_shapefile(rgi_raw_dir)
    rgi_mask, glacier_catalog = rasterize_rgi_mask(shp_path, bbox, (h_grid, w_grid), common_transform, target_crs)

    total_pixels = h_grid * w_grid
    rgi_glacier_pixels = int(np.sum(rgi_mask))
    print(f"[+] Ground Truth Mask Generated: {rgi_glacier_pixels:,} glacier pixels ({rgi_glacier_pixels / total_pixels * 100:.2f}% of basin)")

    # Combined valid mask
    combined_valid = valid_mask_s2 & (l8_bands_data[0] > 0.0)

    # 6. Extract Spatially Paired Patches (256x256 with stride)
    print(f"\n[4/5] Extracting spatially matched {patch_size}x{patch_size} patches (stride={stride})...")
    row_starts = []
    r_curr = 0
    while r_curr + patch_size <= h_grid:
        row_starts.append(r_curr)
        r_curr += stride
    if row_starts[-1] + patch_size < h_grid:
        row_starts.append(h_grid - patch_size)

    col_starts = []
    c_curr = 0
    while c_curr + patch_size <= w_grid:
        col_starts.append(c_curr)
        c_curr += stride
    if col_starts[-1] + patch_size < w_grid:
        col_starts.append(w_grid - patch_size)

    saved_count = 0
    patch_catalog = []

    for r_start in row_starts:
        for c_start in col_starts:
            r_end = r_start + patch_size
            c_end = c_start + patch_size

            patch_valid = combined_valid[r_start:r_end, c_start:c_end]
            valid_ratio = float(np.mean(patch_valid))

            # Skip patch if heavy cloud or no-data
            if valid_ratio < 0.65:
                continue

            s2_sub = s2_8ch[:, r_start:r_end, c_start:c_end].astype(np.float32)
            l8_sub = l8_8ch[:, r_start:r_end, c_start:c_end].astype(np.float32)
            mask_sub = rgi_mask[r_start:r_end, c_start:c_end].astype(np.float32)

            pid = f"patch_{saved_count:03d}"
            s2_pfile = os.path.join(s2_dir, f"{pid}_patch.npy")
            l8_pfile = os.path.join(l8_dir, f"{pid}_patch.npy")
            mask_pfile = os.path.join(mask_dir, f"{pid}_mask.npy")

            np.save(s2_pfile, s2_sub)
            np.save(l8_pfile, l8_sub)
            np.save(mask_pfile, mask_sub)

            ice_pixels = int(np.sum(mask_sub))
            ice_pct = (ice_pixels / (patch_size * patch_size)) * 100.0

            # Compute geographic corner coordinates of this patch
            px_min = snap_xmin + c_start * resolution
            px_max = snap_xmin + c_end * resolution
            py_max = snap_ymax - r_start * resolution
            py_min = snap_ymax - r_end * resolution

            patch_catalog.append({
                "patch_id": pid,
                "row_range": [r_start, r_end],
                "col_range": [c_start, c_end],
                "utm_extent": [px_min, py_min, px_max, py_max],
                "ice_pixels": ice_pixels,
                "ice_percentage": round(ice_pct, 2),
                "valid_pixel_ratio": round(valid_ratio, 3),
                "mean_s2_ndsi": float(np.mean(s2_sub[6])),
                "mean_l8_ndsi": float(np.mean(l8_sub[6])),
            })
            saved_count += 1

    # 7. Export Complete Acquisition & Harmonization Metadata
    metadata = {
        "harmonization_standard": "NASA Harmonized Landsat-Sentinel (HLS) Common 30m Grid",
        "study_area": "Gangotri Glacier Basin, Uttarakhand, India",
        "study_area_wgs84_bbox": bbox,
        "utm_crs": target_crs,
        "common_pixel_resolution_meters": resolution,
        "harmonized_grid_shape": [h_grid, w_grid],
        "affine_transform": [common_transform.a, common_transform.b, common_transform.c,
                             common_transform.d, common_transform.e, common_transform.f],
        "sentinel2_metadata": {
            "scene_id": s2_id,
            "datetime": s2_datetime,
            "cloud_cover_percentage": s2_cloud,
            "source": "AWS Earth Search COG Archive (Public HTTPS)",
            "real_data_acquired": True,
            "bands_acquired": ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
            "resampling_to_30m": "Bilinear interpolation"
        },
        "landsat_metadata": {
            "scene_id": l8_id,
            "datetime": l8_datetime,
            "cloud_cover_percentage": l8_cloud,
            "source": "Microsoft Planetary Computer STAC (SAS Signed HTTPS)",
            "real_data_acquired": True,
            "bands_acquired": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            "resampling_to_30m": "Bilinear interpolation"
        },
        "temporal_delta_days": int(abs((np.datetime64(s2_datetime[:10]) - np.datetime64(l8_datetime[:10])).astype(int))),
        "rgi_ground_truth": {
            "inventory_version": "Randolph Glacier Inventory 6.0",
            "region": "Region 15: South Asia East (Garhwal Himalaya)",
            "primary_glacier": "Gangotri Glacier (RGI60-15.06881)",
            "total_glaciers_intersecting": len(glacier_catalog),
            "glacier_pixels_30m": rgi_glacier_pixels,
            "glacier_coverage_basin_pct": round(rgi_glacier_pixels / total_pixels * 100.0, 2)
        },
        "patch_extraction": {
            "patch_size": patch_size,
            "stride": stride,
            "total_patches_extracted": saved_count
        },
        "catalog": patch_catalog
    }

    meta_path = os.path.join(output_dir, "acquisition_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[5/5] Acquisition & Spatial Harmonization Complete:")
    print(f"      Total Matched 30m Patches Extracted: {saved_count}")
    print(f"      Sentinel-2 Directory: {s2_dir}")
    print(f"      Landsat Directory:    {l8_dir}")
    print(f"      RGI Masks Directory:  {mask_dir}")
    print(f"      Metadata Log:         {meta_path}")
    print("=" * 80)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Multi-Satellite Data Acquisition & Spatial Harmonization")
    parser.add_argument("--bbox", nargs=4, type=float, default=[79.05, 30.80, 79.30, 31.05],
                        help="Bounding box [min_lon, min_lat, max_lon, max_lat]")
    parser.add_argument("--date-range", type=str, default="2024-09-01T00:00:00Z/2024-10-31T23:59:59Z",
                        help="Temporal window for ablation season")
    parser.add_argument("--output-dir", type=str, default="data/satellite_patches",
                        help="Output directory for extracted patches")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size in pixels")
    parser.add_argument("--stride", type=int, default=64,
                        help="Sliding window stride in pixels")
    parser.add_argument("--max-cloud", type=float, default=20.0,
                        help="Maximum cloud cover percentage")
    args = parser.parse_args()

    acquire_harmonized_dataset(
        bbox=args.bbox,
        date_range=args.date_range,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        max_cloud=args.max_cloud
    )


if __name__ == "__main__":
    main()
