"""
Sentinel-2 Data Acquisition Script for Gangotri Glacier Basin
Queries AWS Earth Search STAC for Sentinel-2 Level-2A Surface Reflectance (Harmonized)
Extracts spectral bands (B02, B03, B04, B08, B11, B12) over MGRS Tile 44RLV.
"""

import os
import sys
import json
import argparse
import requests
import rasterio

# Ensure repo root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Study Area: Gangotri Glacier Basin, Uttarakhand, India
DEFAULT_BBOX = [78.95, 30.70, 79.45, 31.15]
DEFAULT_DATE_RANGE = "2024-09-01T00:00:00Z/2024-11-30T23:59:59Z"


def search_sentinel2_scenes(bbox=None, date_range=None, max_cloud=15.0, limit=10):
    bbox = bbox or DEFAULT_BBOX
    date_range = date_range or DEFAULT_DATE_RANGE
    url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "bbox": bbox,
        "datetime": date_range,
        "collections": ["sentinel-2-l2a"],
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    print(f"[*] Querying AWS Earth Search STAC for Sentinel-2 L2A scenes in {date_range}...")
    resp = requests.post(url, json=payload, timeout=25)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    print(f"[+] Found {len(features)} matching Sentinel-2 Level-2A scenes (<{max_cloud}% cloud).")
    return features


def main():
    parser = argparse.ArgumentParser(description="Query and Inspect Sentinel-2 Level-2A Scenes")
    parser.add_argument("--date-range", type=str, default=DEFAULT_DATE_RANGE, help="ISO8601 date range")
    parser.add_argument("--max-cloud", type=float, default=15.0, help="Maximum cloud cover %")
    parser.add_argument("--limit", type=int, default=5, help="Number of scenes to return")
    parser.add_argument("--output-json", type=str, default="data/sentinel2/s2_search_scenes.json")
    args = parser.parse_args()

    scenes = search_sentinel2_scenes(date_range=args.date_range, max_cloud=args.max_cloud, limit=args.limit)

    for i, s in enumerate(scenes):
        props = s.get("properties", {})
        print(f"[{i+1}] Scene ID: {s.get('id')} | Date: {props.get('datetime', '')[:10]} | Cloud: {props.get('eo:cloud_cover', 0):.2f}%")

    out_path = os.path.join(repo_root, args.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"[+] Saved scenes metadata to {out_path}")


if __name__ == "__main__":
    main()
