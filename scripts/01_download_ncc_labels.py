"""Download the official NCC 2050 SSP2-4.5 1% AEP coastal polygons.

This uses only Python's standard library, so it can run before geospatial
packages are installed. Feature attributes are retained exactly as published.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "labels"
SERVICE = (
    "https://services1.arcgis.com/Y4k7lyf2XTGeQC6V/ArcGIS/rest/services/"
    "2050_SSP5_4_5/FeatureServer/4/query"
)


def feature_collection(features: list[dict], crs: dict | None = None) -> dict:
    result = {"type": "FeatureCollection", "features": features}
    if crs:
        result["crs"] = crs
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "2193",
        "f": "geojson",
    }
    url = SERVICE + "?" + urlencode(params)
    print("Downloading NCC label polygons...")
    with urlopen(url, timeout=120) as response:
        raw = json.load(response)

    features = raw.get("features", [])
    if not features:
        raise RuntimeError("The service returned no features.")

    connected = [f for f in features if f.get("properties", {}).get("connected") == 1]
    tahunanui = [
        f
        for f in connected
        if f.get("properties", {}).get("InundationCellName") == "Tahunanui"
    ]
    if not tahunanui:
        raise RuntimeError("Tahunanui connected polygon was not found.")

    crs = raw.get("crs")
    write_json(OUT / "ncc_coastal_2050_ssp2_4_5_1pct_all.geojson", raw)
    write_json(
        OUT / "ncc_coastal_2050_ssp2_4_5_1pct_connected.geojson",
        feature_collection(connected, crs),
    )
    write_json(
        OUT / "ncc_coastal_2050_ssp2_4_5_1pct_tahunanui_connected.geojson",
        feature_collection(tahunanui, crs),
    )
    metadata = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "service": SERVICE.rsplit("/query", 1)[0],
        "query_url": url,
        "all_feature_count": len(features),
        "connected_feature_count": len(connected),
        "tahunanui_connected_feature_count": len(tahunanui),
        "target_definition": "connected=1; timeframe=2050; ssp=SSP2-4.5; aep=1% AEP",
        "warning": (
            "This is modelled screening data. A model trained on it approximates "
            "the published hazard layer; it is not an independent flood forecast."
        ),
    }
    write_json(OUT / "ncc_label_metadata.json", metadata)

    names = sorted({f["properties"]["InundationCellName"] for f in connected})
    print(f"Saved {len(features)} total features and {len(connected)} connected features.")
    print("Coastal cells:", ", ".join(names))
    print("Tahunanui holdout label is ready.")


if __name__ == "__main__":
    main()
