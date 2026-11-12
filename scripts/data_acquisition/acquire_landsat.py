"""
Landsat 8/9 Data Acquisition Script for Gangotri Glacier Basin
Queries Microsoft Planetary Computer STAC for Landsat Collection 2 Level-2 Surface Reflectance.
Signs asset URLs with public SAS tokens and retrieves spectral bands over Gangotri Glacier.
"""

import os
import sys
import json
import argparse
import requests

# Ensure repo root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

DEFAULT_BBOX = [78.95, 30.70, 79.45, 31.15]
DEFAULT_DATE_RANGE = "2024-09-01T00:00:00Z/2024-11-30T23:59:59Z"


def search_landsat_scenes(bbox=None, date_range=None, max_cloud=15.0, limit=5):
    bbox = bbox or DEFAULT_BBOX
    date_range = date_range or DEFAULT_DATE_RANGE
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "bbox": bbox,
        "datetime": date_range,
        "collections": ["landsat-c2-l2"],
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    print(f"[*] Querying Microsoft Planetary Computer STAC for Landsat 8/9 Level-2 in {date_range}...")
    resp = requests.post(url, json=payload, timeout=25)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    print(f"[+] Found {len(features)} matching Landsat Collection 2 Level-2 scenes (<{max_cloud}% cloud).")
    return features


def sign_planetary_computer_url(asset_href: str) -> str:
    sign_url = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={asset_href}"
    resp = requests.get(sign_url, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("href", asset_href)
    return asset_href


def main():
    parser = argparse.ArgumentParser(description="Query Landsat 8/9 Collection 2 Level-2 Scenes")
    parser.add_argument("--date-range", type=str, default=DEFAULT_DATE_RANGE, help="ISO8601 date range")
    parser.add_argument("--max-cloud", type=float, default=15.0, help="Maximum cloud cover %")
    parser.add_argument("--limit", type=int, default=5, help="Number of scenes to return")
    parser.add_argument("--output-json", type=str, default="data/landsat/landsat_search_scenes.json")
    args = parser.parse_args()

    scenes = search_landsat_scenes(date_range=args.date_range, max_cloud=args.max_cloud, limit=args.limit)

    for i, s in enumerate(scenes):
        props = s.get("properties", {})
        platform = props.get("platform", "Landsat")
        print(f"[{i+1}] Platform: {platform} | ID: {s.get('id')} | Date: {props.get('datetime', '')[:10]} | Cloud: {props.get('eo:cloud_cover', 0):.2f}%")

    out_path = os.path.join(repo_root, args.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"[+] Saved Landsat metadata to {out_path}")


if __name__ == "__main__":
    main()
