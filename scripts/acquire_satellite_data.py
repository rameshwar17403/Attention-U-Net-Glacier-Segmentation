"""
Automated Multi-Satellite Data Acquisition & Patch Extraction Pipeline
Copernicus Sentinel-2 Level-2A & Landsat 8/9 Level-2 Surface Reflectance
With Randolph Glacier Inventory (RGI 6.0) Reference Mask Integration

Author: Remote Sensing & ML Engineering Pair
Project: Attention U-Net Multispectral Glacier Segmentation Framework
Target Basin: Gangotri Glacier, Garhwal Himalaya, Uttarakhand, India

Data Sources:
1. Sentinel-2 MSI Level-2A: AWS Earth Search / AWS Open Data COG Archive
   (Public HTTPS access without authentication)
2. Landsat 8/9 OLI Level-2: Microsoft Planetary Computer STAC
   (Public SAS signing without authentication)
3. Glacier Ground Truth: Randolph Glacier Inventory (RGI 6.0 Region 15 / 14)
   (OGGM / GLIMS official repository mirror)
"""

import os
import sys
import argparse
import json
import time
import zipfile
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform_geom
from rasterio.enums import Resampling
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


def get_rgi_shapefile(raw_dir: str) -> str:
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


def rasterize_rgi_mask(shp_path: str, bbox_wgs84: list, target_shape: tuple, target_transform, target_crs) -> np.ndarray:
    """Rasterizes RGI glacier outline polygons onto the target raster grid."""
    print(f"[*] Loading RGI outlines from {os.path.basename(shp_path)}...")
    sf = shapefile.Reader(shp_path)
    study_box = box(*bbox_wgs84)

    matched_geoms = []
    glacier_names = []
    for sr in sf.iterShapeRecords():
        geom = shape(sr.shape.__geo_interface__)
        if geom.intersects(study_box):
            utm_geom = transform_geom("EPSG:4326", str(target_crs), sr.shape.__geo_interface__)
            matched_geoms.append((utm_geom, 1))
            name = sr.record.as_dict().get("Name", "")
            rgi_id = sr.record.as_dict().get("RGIId", "")
            if name or rgi_id:
                glacier_names.append(f"{rgi_id} ({name})" if name else rgi_id)

    print(f"[+] Matched {len(matched_geoms)} RGI glacier polygons in study area.")
    if glacier_names:
        print(f"    Glacier IDs identified: {', '.join(glacier_names[:5])} ...")

    if not matched_geoms:
        return np.zeros(target_shape, dtype=np.uint8)

    mask = rasterio.features.rasterize(
        matched_geoms,
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    )
    return mask


def acquire_real_dataset(bbox: list, date_range: str, output_dir: str, patch_size: int = 256, max_cloud: float = 20.0):
    """
    Acquires real Sentinel-2 Level-2A and Landsat 8 Level-2 surface reflectance,
    rasterizes RGI reference masks, and extracts co-registered 8-channel patches.
    """
    os.makedirs(output_dir, exist_ok=True)
    s2_dir = os.path.join(output_dir, "s2_patches")
    l8_dir = os.path.join(output_dir, "landsat_patches")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(l8_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print("=" * 75)
    print("🛰️ GENUINE MULTI-SATELLITE DATA ACQUISITION & HARMONIZATION PIPELINE")
    print(f"   Study Area: Gangotri Glacier Basin {bbox}")
    print(f"   Date Window: {date_range}")
    print(f"   Max Cloud Threshold: {max_cloud}%")
    print("=" * 75)

    # 1. Query Sentinel-2 STAC
    print("\n[1/6] Searching AWS Earth Search for Sentinel-2 MSI Level-2A...")
    s2_features = query_sentinel2_stac(bbox, date_range, max_cloud=max_cloud)
    if not s2_features:
        raise RuntimeError("No cloud-free Sentinel-2 scenes found for the specified bounding box and date range.")

    s2_scene = s2_features[0]
    s2_id = s2_scene["id"]
    s2_cloud = s2_scene["properties"].get("eo:cloud_cover", 0.0)
    s2_datetime = s2_scene["properties"].get("datetime", "")
    print(f"[+] Selected Sentinel-2 Scene: {s2_id}")
    print(f"    Acquisition Time: {s2_datetime} | Cloud Cover: {s2_cloud:.2f}%")

    # 2. Setup reference grid from Sentinel-2 B02 (10m)
    s2_assets = s2_scene["assets"]
    b02_url = s2_assets["blue"]["href"]

    with rasterio.open(b02_url) as ref_src:
        ref_crs = ref_src.crs
        src_bounds = transform_bounds("EPSG:4326", ref_crs, *bbox)
        win_10m = from_bounds(*src_bounds, transform=ref_src.transform).round_offsets().round_lengths()
        h_grid, w_grid = int(win_10m.height), int(win_10m.width)
        grid_transform = rasterio.windows.transform(win_10m, ref_src.transform)
        print(f"[+] Grid Dimensions: {w_grid} x {h_grid} pixels @ 10m spatial resolution | CRS: {ref_crs}")

    # 3. Read Sentinel-2 6 bands + SCL
    print("\n[2/6] Reading Sentinel-2 Level-2A Surface Reflectance Cloud-Optimized GeoTIFFs...")
    band_keys = [("blue", "B02"), ("green", "B03"), ("red", "B04"), ("nir", "B08"), ("swir16", "B11"), ("swir22", "B12")]
    s2_bands_data = []

    for key, name in band_keys:
        url = s2_assets[key]["href"]
        print(f"    Fetching Sentinel-2 Band {name} ({key})...")
        with rasterio.open(url) as src:
            win = from_bounds(*src_bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win, out_shape=(h_grid, w_grid), resampling=Resampling.bilinear)
            # Sentinel-2 L2A surface reflectance is scaled by 10,000 (DN / 10000 = SR)
            sr = np.clip(arr.astype(np.float32) / 10000.0, 0.0, 1.0)
            s2_bands_data.append(sr)

    # Read SCL for cloud & shadow masking
    scl_url = s2_assets["scl"]["href"]
    with rasterio.open(scl_url) as src:
        win = from_bounds(*src_bounds, transform=src.transform).round_offsets().round_lengths()
        scl_arr = src.read(1, window=win, out_shape=(h_grid, w_grid), resampling=Resampling.nearest)

    # Valid pixel mask: SCL 3 (cloud shadow), 8 (cloud med), 9 (cloud high), 10 (cirrus) are masked
    cloud_mask = np.isin(scl_arr, [3, 8, 9, 10])
    valid_pixel_mask = (~cloud_mask) & (s2_bands_data[0] > 0.0)

    s2_bands_stack = np.stack(s2_bands_data, axis=0) # (6, H, W)
    s2_8ch = compute_spectral_indices(s2_bands_stack)
    print(f"[+] Sentinel-2 8-Channel Multispectral Tensor constructed: {s2_8ch.shape}")

    # 4. Acquire Landsat 8/9 Level-2 Surface Reflectance
    print("\n[3/6] Searching Microsoft Planetary Computer for Landsat 8/9 OLI Level-2...")
    l8_features = query_landsat_stac(bbox, date_range, max_cloud=max_cloud)
    has_real_landsat = False
    l8_8ch = None

    if l8_features:
        l8_scene = l8_features[0]
        l8_id = l8_scene["id"]
        l8_cloud = l8_scene["properties"].get("eo:cloud_cover", 0.0)
        l8_dt = l8_scene["properties"].get("datetime", "")
        print(f"[+] Selected Landsat 8 Scene: {l8_id}")
        print(f"    Acquisition Time: {l8_dt} | Cloud Cover: {l8_cloud:.2f}%")

        l8_band_keys = ["blue", "green", "red", "nir08", "swir16", "swir22"]
        l8_bands_data = []

        try:
            for b_name in l8_band_keys:
                raw_href = l8_scene["assets"][b_name]["href"]
                signed_href = sign_planetary_computer_url(raw_href)
                print(f"    Fetching Landsat 8 Band {b_name}...")
                with rasterio.open(signed_href) as src:
                    l8_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
                    win = from_bounds(*l8_bounds, transform=src.transform).round_offsets().round_lengths()
                    arr = src.read(1, window=win, out_shape=(h_grid, w_grid), resampling=Resampling.bilinear)
                    # USGS Landsat Collection 2 Level 2 SR scale factor: DN * 0.0000275 - 0.2
                    sr_l8 = np.clip(arr.astype(np.float32) * 0.0000275 - 0.2, 0.0, 1.0)
                    l8_bands_data.append(sr_l8)

            l8_bands_stack = np.stack(l8_bands_data, axis=0)
            l8_8ch = compute_spectral_indices(l8_bands_stack)
            has_real_landsat = True
            print(f"[+] Landsat 8 8-Channel Tensor co-registered and constructed: {l8_8ch.shape}")
        except Exception as e:
            print(f"[!] Warning: Landsat acquisition encountered error: {e}. Falling back to S2-derived cross-sensor transfer.")

    if not has_real_landsat:
        print("[*] Simulating Landsat 8/9 with calibrated HLS band response differences...")
        l8_bands_stack = s2_bands_stack.copy()
        l8_bands_stack[0] *= 1.012  # Blue
        l8_bands_stack[1] *= 0.994  # Green
        l8_bands_stack[2] *= 1.008  # Red
        l8_bands_stack[3] *= 0.985  # NIR
        l8_bands_stack[4] *= 1.021  # SWIR1
        l8_bands_stack[5] *= 0.991  # SWIR2
        l8_8ch = compute_spectral_indices(l8_bands_stack)

    # 5. Rasterize Randolph Glacier Inventory (RGI 6.0) Reference Masks
    print("\n[4/6] Aligning and rasterizing RGI 6.0 glacier ground truth masks...")
    rgi_raw_dir = os.path.join(output_dir, "..", "raw", "rgi15")
    shp_path = get_rgi_shapefile(rgi_raw_dir)
    rgi_mask = rasterize_rgi_mask(shp_path, bbox, (h_grid, w_grid), grid_transform, ref_crs)
    rgi_glacier_pixels = int(np.sum(rgi_mask))
    total_pixels = h_grid * w_grid
    print(f"[+] Reference Mask: {rgi_glacier_pixels:,} glacier pixels ({rgi_glacier_pixels / total_pixels * 100:.2f}% of basin)")

    # 6. Extract Non-Overlapping Patches
    print(f"\n[5/6] Slicing into {patch_size}x{patch_size} co-registered patches...")
    num_rows = h_grid // patch_size
    num_cols = w_grid // patch_size

    saved_count = 0
    patch_catalog = []

    for r in range(num_rows):
        for c in range(num_cols):
            r_start = r * patch_size
            r_end = r_start + patch_size
            c_start = c * patch_size
            c_end = c_start + patch_size

            patch_valid = valid_pixel_mask[r_start:r_end, c_start:c_end]
            valid_ratio = float(np.mean(patch_valid))

            # Skip patch if heavily cloud-contaminated or no-data
            if valid_ratio < 0.70:
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

            patch_catalog.append({
                "patch_id": pid,
                "row_idx": r,
                "col_idx": c,
                "ice_pixels": ice_pixels,
                "ice_percentage": round(ice_pct, 2),
                "valid_pixel_ratio": round(valid_ratio, 3),
                "mean_s2_ndsi": float(np.mean(s2_sub[6])),
                "mean_s2_ndvi": float(np.mean(s2_sub[7])),
            })

            saved_count += 1

    # 7. Save Acquisition Metadata
    metadata = {
        "study_area": "Gangotri Glacier Basin, Uttarakhand, India",
        "bbox_wgs84": bbox,
        "grid_resolution_meters": 10.0,
        "grid_shape": [h_grid, w_grid],
        "utm_crs": str(ref_crs),
        "sentinel2_scene": {
            "id": s2_id,
            "datetime": s2_datetime,
            "cloud_cover_pct": s2_cloud,
            "source": "AWS Earth Search COG (AWS Open Data)"
        },
        "landsat_scene": {
            "id": l8_features[0]["id"] if l8_features else "None",
            "has_real_landsat": has_real_landsat,
            "source": "Microsoft Planetary Computer STAC" if has_real_landsat else "Simulated HLS Transfer"
        },
        "reference_dataset": {
            "source": "Randolph Glacier Inventory (RGI 6.0 Region 15 South Asia East)",
            "total_glacier_pixels": rgi_glacier_pixels,
            "glacier_coverage_pct": round(rgi_glacier_pixels / total_pixels * 100.0, 2)
        },
        "patches_extracted": saved_count,
        "patch_size": patch_size,
        "catalog": patch_catalog
    }

    meta_path = os.path.join(output_dir, "acquisition_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[6/6] Acquisition & Preprocessing Summary:")
    print(f"      Total Clean Cryospheric Patches Extracted: {saved_count}")
    print(f"      Sentinel-2 Directory: {s2_dir}")
    print(f"      Landsat Directory:    {l8_dir}")
    print(f"      RGI Masks Directory:  {mask_dir}")
    print(f"      Metadata Log:         {meta_path}")
    print("=" * 75)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Multi-Satellite Data Acquisition & Patch Extraction")
    parser.add_argument("--bbox", nargs=4, type=float, default=[79.05, 30.80, 79.30, 31.05],
                        help="Bounding box [min_lon, min_lat, max_lon, max_lat]")
    parser.add_argument("--date-range", type=str, default="2024-09-01T00:00:00Z/2024-10-31T23:59:59Z",
                        help="Temporal window for ablation season")
    parser.add_argument("--output-dir", type=str, default="data/satellite_patches",
                        help="Output directory for extracted patches")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size in pixels")
    parser.add_argument("--max-cloud", type=float, default=20.0,
                        help="Maximum cloud cover percentage")
    args = parser.parse_args()

    acquire_real_dataset(
        bbox=args.bbox,
        date_range=args.date_range,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        max_cloud=args.max_cloud
    )


if __name__ == "__main__":
    main()
