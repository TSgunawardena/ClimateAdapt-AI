from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_PATH = PROJECT_ROOT / "data/raw/dem/BQ26.tif"
LAND_COVER_PATH = PROJECT_ROOT / "data/raw/land_cover/lcdb_v6_nelson.gpkg"
COASTLINE_PATH = PROJECT_ROOT / "data/raw/coastline/nz_coastline_topo50.gpkg"
LABEL_PATH = (
    PROJECT_ROOT
    / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_tahunanui_connected.geojson"
)


def inspect_raster(path):
    print("\n" + "=" * 70)
    print(f"DEM: {path.name}")
    print("=" * 70)

    with rasterio.open(path) as src:
        print("CRS:", src.crs)
        print("Width x height:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Number of bands:", src.count)
        print("Data type:", src.dtypes[0])
        print("NoData value:", src.nodata)
        print("Bounds:", src.bounds)

        dem_bounds = box(
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
        )

        return src.crs, dem_bounds


def inspect_vector(name, path):
    print("\n" + "=" * 70)
    print(f"{name}: {path.name}")
    print("=" * 70)

    layers = gpd.list_layers(path)
    print("Available layers:")
    print(layers.to_string(index=False))

    layer_name = layers.iloc[0]["name"]
    data = gpd.read_file(path, layer=layer_name)

    print("\nSelected layer:", layer_name)
    print("CRS:", data.crs)
    print("Feature count:", len(data))
    print("Geometry types:", data.geometry.geom_type.unique().tolist())
    print("Bounds:", data.total_bounds)
    print("Columns:", data.columns.tolist())

    return data


dem_crs, dem_bounds = inspect_raster(DEM_PATH)

land_cover = inspect_vector("LAND COVER", LAND_COVER_PATH)
coastline = inspect_vector("COASTLINE", COASTLINE_PATH)
labels = inspect_vector("TAHUNANUI FLOOD LABEL", LABEL_PATH)


print("\n" + "=" * 70)
print("DATA COMPATIBILITY CHECK")
print("=" * 70)

datasets = {
    "Land cover": land_cover,
    "Coastline": coastline,
    "Flood label": labels,
}

for name, data in datasets.items():
    crs_match = data.crs == dem_crs
    print(f"{name} CRS matches DEM: {crs_match}")


labels_in_dem_crs = labels.to_crs(dem_crs)
label_geometry = labels_in_dem_crs.geometry.union_all()

original_label_area = label_geometry.area
covered_geometry = label_geometry.intersection(dem_bounds)
covered_area = covered_geometry.area

if original_label_area > 0:
    coverage_percentage = (covered_area / original_label_area) * 100
else:
    coverage_percentage = 0

print(f"Flood-label area covered by DEM: {coverage_percentage:.2f}%")

if coverage_percentage >= 99:
    print("RESULT: The DEM provides complete coverage of the flood label.")
elif coverage_percentage > 0:
    print("WARNING: The DEM covers only part of the flood label.")
else:
    print("ERROR: The DEM and flood label do not overlap.")

print("\nInspection completed.")