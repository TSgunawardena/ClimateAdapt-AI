from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHECKS = {
    "NCC all labels": ROOT / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_all.geojson",
    "NCC connected labels": ROOT / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_connected.geojson",
    "Tahunanui test label": ROOT / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_tahunanui_connected.geojson",
    "LCDB land cover": ROOT / "data/raw/land_cover/lcdb_v6_nelson.gpkg",
    "Coastline": ROOT / "data/raw/coastline/nz_coastline_topo50.gpkg",
}

for name, path in CHECKS.items():
    state = "READY" if path.exists() else "MISSING"
    print(f"{state:7}  {name:23}  {path.relative_to(ROOT)}")

dem_files = sorted((ROOT / "data/raw/dem").glob("*.tif"))
dem_state = "READY" if dem_files else "MISSING"
dem_value = ", ".join(p.name for p in dem_files) if dem_files else "data/raw/dem/*.tif"
print(f"{dem_state:7}  {'DEM crop(s)':23}  {dem_value}")
