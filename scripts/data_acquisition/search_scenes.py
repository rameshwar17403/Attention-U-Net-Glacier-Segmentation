"""
Searches AWS Earth Search STAC for Sentinel-2 L2A and Planetary Computer for Landsat 8/9
over Gangotri Glacier basin across multiple post-monsoon ablation seasons (2018-2024).
"""
import os
import json
import requests

BBOX = [79.05, 30.80, 79.30, 31.05]
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

def search_s2_for_year(year):
    url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "bbox": BBOX,
        "datetime": f"{year}-09-15T00:00:00Z/{year}-11-15T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "query": {"eo:cloud_cover": {"lt": 15}},
        "limit": 15,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    try:
        r = requests.post(url, json=payload, timeout=25)
        if r.status_code == 200:
            return r.json().get("features", [])
    except Exception as e:
        print(f"[!] Error querying S2 for {year}: {e}")
    return []

def search_landsat_for_year(year):
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "bbox": BBOX,
        "datetime": f"{year}-09-15T00:00:00Z/{year}-11-15T23:59:59Z",
        "collections": ["landsat-c2-l2"],
        "query": {"eo:cloud_cover": {"lt": 15}},
        "limit": 10,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    try:
        r = requests.post(url, json=payload, timeout=25)
        if r.status_code == 200:
            return r.json().get("features", [])
    except Exception as e:
        print(f"[!] Error querying Landsat for {year}: {e}")
    return []

def main():
    print("================================================================================")
    print("🛰️ MULTI-YEAR SCENE INVENTORY SEARCH (GANGOTRI GLACIER BASIN)")
    print(f"   BBox: {BBOX} | Temporal Window: Sept 15 - Nov 15 (2018-2024)")
    print("================================================================================")

    all_s2_scenes = {}
    all_landsat_scenes = {}

    for yr in YEARS:
        s2_items = search_s2_for_year(yr)
        ls_items = search_landsat_for_year(yr)

        print(f"\n--- Year {yr} ---")
        print(f"Sentinel-2 Scenes Found (<15% cloud): {len(s2_items)}")
        for item in s2_items[:3]:
            dt = item["properties"].get("datetime")
            cloud = item["properties"].get("eo:cloud_cover", 0.0)
            print(f"   S2: {item['id']} | Date: {dt[:10]} | Cloud: {cloud:.2f}%")
        if s2_items:
            all_s2_scenes[str(yr)] = s2_items

        print(f"Landsat 8/9 Scenes Found (<15% cloud): {len(ls_items)}")
        for item in ls_items[:3]:
            dt = item["properties"].get("datetime")
            cloud = item["properties"].get("eo:cloud_cover", 0.0)
            print(f"   L8/9: {item['id']} | Date: {dt[:10]} | Cloud: {cloud:.2f}%")
        if ls_items:
            all_landsat_scenes[str(yr)] = ls_items

    # Save search results to disk
    os.makedirs("data/sentinel2", exist_ok=True)
    os.makedirs("data/landsat", exist_ok=True)

    with open("data/sentinel2/stac_search_results.json", "w") as f:
        json.dump(all_s2_scenes, f, indent=2)

    with open("data/landsat/stac_search_results.json", "w") as f:
        json.dump(all_landsat_scenes, f, indent=2)

    print(f"\n[+] Saved S2 candidate scenes to data/sentinel2/stac_search_results.json")
    print(f"[+] Saved Landsat candidate scenes to data/landsat/stac_search_results.json")

if __name__ == "__main__":
    main()
