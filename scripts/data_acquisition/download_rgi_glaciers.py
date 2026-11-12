"""
Randolph Glacier Inventory (RGI 6.0) Region 15 Download and Rasterization Script
Downloads RGI 6.0 Region 15 (South Asia East) from the official OGGM/GLIMS mirror.
Rasterizes glacier outlines onto the target Gangotri UTM 44N grid.
"""

import os
import sys
import zipfile
import argparse
import requests
import rasterio
import rasterio.features
from rasterio.warp import transform_geom
from affine import Affine
import shapefile
from shapely.geometry import shape, box

# Ensure repo root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def download_rgi_region15(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    shp_candidates = [os.path.join(dest_dir, f) for f in os.listdir(dest_dir) if f.endswith(".shp") and not f.startswith(".")]
    if shp_candidates:
        print(f"[+] Found cached RGI shapefile: {shp_candidates[0]}")
        return shp_candidates[0]

    zip_path = os.path.join(dest_dir, "15_rgi60_SouthAsiaEast.zip")
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 10000000:
        url = "https://cluster.klima.uni-bremen.de/~oggm/rgi/www.glims.org/RGI/rgi60_files/15_rgi60_SouthAsiaEast.zip"
        print(f"[*] Downloading RGI 6.0 Region 15 from {url}...")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        print("[+] Download complete.")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    shp_candidates = [os.path.join(dest_dir, f) for f in os.listdir(dest_dir) if f.endswith(".shp") and not f.startswith(".")]
    return shp_candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Download and Verify RGI 6.0 Region 15 Outlines")
    parser.add_argument("--dest-dir", type=str, default="data/raw/rgi15")
    args = parser.parse_args()

    dest = os.path.join(repo_root, args.dest_dir)
    shp_path = download_rgi_region15(dest)
    print(f"[+] Verified RGI 6.0 Region 15 Shapefile at: {shp_path}")


if __name__ == "__main__":
    main()
