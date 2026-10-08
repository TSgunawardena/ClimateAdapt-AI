from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_FOLDER = PROJECT_ROOT / "data/raw/dem"
LABEL_PATH = (
    PROJECT_ROOT
    / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_connected.geojson"
)

dem_files = sorted(DEM_FOLDER.glob("*.tif"))

if not dem_files:
    raise FileNotFoundError("No DEM GeoTIFF files were found.")

print("\nDEM files found:")

dem_boundaries = []
dem_crs = None

for dem_file in dem_files:
    with rasterio.open(dem_file) as source:
        if dem_crs is None:
            dem_crs = source.crs
        elif source.crs != dem_crs:
            raise ValueError(
                f"CRS mismatch: {dem_file.name} uses {source.crs}, "
                f"but the first DEM uses {dem_crs}."
            )

        boundary = box(
            source.bounds.left,
            source.bounds.bottom,
            source.bounds.right,
            source.bounds.top,
        )

        dem_boundaries.append(boundary)

        print(
            f"  {dem_file.name}: "
            f"{source.width} x {source.height}, "
            f"resolution={source.res}, CRS={source.crs}"
        )

combined_dem_boundary = unary_union(dem_boundaries)

labels = gpd.read_file(LABEL_PATH).to_crs(dem_crs)

coastal_areas = labels.dissolve(
    by="InundationCellName",
    as_index=False,
)

results = []

for _, row in coastal_areas.iterrows():
    area_name = row["InundationCellName"]
    geometry = row.geometry

    total_area = geometry.area
    covered_area = geometry.intersection(combined_dem_boundary).area

    coverage_percentage = (
        (covered_area / total_area) * 100
        if total_area > 0
        else 0
    )

    if coverage_percentage >= 95:
        status = "FULL"
    elif coverage_percentage >= 5:
        status = "PARTIAL"
    else:
        status = "NONE"

    results.append(
        (
            area_name,
            coverage_percentage,
            status,
        )
    )

results.sort(key=lambda item: item[1], reverse=True)

print("\n" + "=" * 76)
print("COMBINED DEM COVERAGE")
print("=" * 76)

for area_name, percentage, status in results:
    print(
        f"{area_name:<28} "
        f"{percentage:>7.2f}%   "
        f"{status}"
    )

print("=" * 76)

full_areas = [
    area_name
    for area_name, percentage, status in results
    if status == "FULL"
]

print("\nFully covered coastal areas:")
for area_name in full_areas:
    print(f"  - {area_name}")