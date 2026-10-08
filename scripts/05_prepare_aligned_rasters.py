from pathlib import Path
import json
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_PATH = PROJECT_ROOT / "data/raw/dem/BQ26.tif"
LAND_COVER_PATH = PROJECT_ROOT / "data/raw/land_cover/lcdb_v6_nelson.gpkg"
COASTLINE_PATH = PROJECT_ROOT / "data/raw/coastline/nz_coastline_topo50.gpkg"
LABEL_PATH = (
    PROJECT_ROOT
    / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_tahunanui_connected.geojson"
)

OUTPUT_FOLDER = PROJECT_ROOT / "data/processed/rasters"
EVIDENCE_FOLDER = PROJECT_ROOT / "evidence/figures"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
EVIDENCE_FOLDER.mkdir(parents=True, exist_ok=True)

TARGET_RESOLUTION = 5.0


# ---------------------------------------------------------
# Helper function for saving GeoTIFF files
# ---------------------------------------------------------
def save_raster(filename, array, transform, crs, dtype, nodata):
    output_path = OUTPUT_FOLDER / filename

    output_array = array.astype(dtype)

    if np.issubdtype(output_array.dtype, np.floating):
        output_array = np.where(
            np.isfinite(output_array),
            output_array,
            nodata,
        )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=output_array.shape[0],
        width=output_array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as destination:
        destination.write(output_array, 1)

    print(f"SAVED: {output_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------
# Step 1: Create the common 5-metre grid
# ---------------------------------------------------------
print("\nCreating the common 5-metre raster grid...")

with rasterio.open(DEM_PATH) as source:
    source_bounds = source.bounds
    source_crs = source.crs

    grid_width = math.ceil(
        (source_bounds.right - source_bounds.left) / TARGET_RESOLUTION
    )
    grid_height = math.ceil(
        (source_bounds.top - source_bounds.bottom) / TARGET_RESOLUTION
    )

    grid_transform = from_origin(
        source_bounds.left,
        source_bounds.top,
        TARGET_RESOLUTION,
        TARGET_RESOLUTION,
    )

    elevation = np.full(
        (grid_height, grid_width),
        np.nan,
        dtype=np.float32,
    )

    reproject(
        source=rasterio.band(source, 1),
        destination=elevation,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=source.nodata,
        dst_transform=grid_transform,
        dst_crs=source_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )

print(f"Grid size: {grid_width} columns x {grid_height} rows")
print(f"Grid resolution: {TARGET_RESOLUTION} metres")


# ---------------------------------------------------------
# Step 2: Load vector datasets
# ---------------------------------------------------------
print("\nLoading vector datasets...")

land_cover = gpd.read_file(LAND_COVER_PATH).to_crs(source_crs)
coastline = gpd.read_file(COASTLINE_PATH).to_crs(source_crs)
flood_label = gpd.read_file(LABEL_PATH).to_crs(source_crs)


# ---------------------------------------------------------
# Step 3: Create land mask
# 1 = land, 0 = water/outside land
# ---------------------------------------------------------
print("Creating land mask...")

land_shapes = [
    (geometry, 1)
    for geometry in coastline.geometry
    if geometry is not None and not geometry.is_empty
]

land_mask = rasterize(
    land_shapes,
    out_shape=(grid_height, grid_width),
    transform=grid_transform,
    fill=0,
    all_touched=True,
    dtype=np.uint8,
)


# ---------------------------------------------------------
# Step 4: Create land-cover raster
# ---------------------------------------------------------
print("Creating land-cover raster...")

land_cover_shapes = []

for _, row in land_cover.iterrows():
    geometry = row.geometry
    class_value = row.get("Class_2023")

    if (
        geometry is not None
        and not geometry.is_empty
        and pd.notna(class_value)
    ):
        land_cover_shapes.append(
            (geometry, int(float(class_value)))
        )

land_cover_raster = rasterize(
    land_cover_shapes,
    out_shape=(grid_height, grid_width),
    transform=grid_transform,
    fill=0,
    all_touched=True,
    dtype=np.uint16,
)

# Land-cover values outside land are not used
land_cover_raster[land_mask == 0] = 0


# ---------------------------------------------------------
# Step 5: Create official flood target
# 1 = flood-susceptible land
# 0 = non-flood-susceptible land
# ---------------------------------------------------------
print("Creating binary flood target...")

flood_shapes = [
    (geometry, 1)
    for geometry in flood_label.geometry
    if geometry is not None and not geometry.is_empty
]

raw_flood_mask = rasterize(
    flood_shapes,
    out_shape=(grid_height, grid_width),
    transform=grid_transform,
    fill=0,
    all_touched=True,
    dtype=np.uint8,
)

# Remove ocean cells from the official flood polygon
flood_target = (
    (raw_flood_mask == 1)
    & (land_mask == 1)
    & np.isfinite(elevation)
).astype(np.uint8)


# ---------------------------------------------------------
# Step 6: Create valid training-area mask
# ---------------------------------------------------------
valid_land_mask = (
    (land_mask == 1)
    & np.isfinite(elevation)
).astype(np.uint8)


# ---------------------------------------------------------
# Step 7: Calculate slope from elevation
# ---------------------------------------------------------
print("Calculating slope...")

finite_elevation = elevation[np.isfinite(elevation)]

if len(finite_elevation) == 0:
    raise ValueError("The processed DEM contains no valid elevation values.")

replacement_value = float(np.median(finite_elevation))

elevation_for_slope = np.where(
    np.isfinite(elevation),
    elevation,
    replacement_value,
)

gradient_y, gradient_x = np.gradient(
    elevation_for_slope,
    TARGET_RESOLUTION,
    TARGET_RESOLUTION,
)

slope_radians = np.arctan(
    np.sqrt((gradient_x ** 2) + (gradient_y ** 2))
)

slope_degrees = np.degrees(slope_radians).astype(np.float32)
slope_degrees[valid_land_mask == 0] = np.nan


# ---------------------------------------------------------
# Step 8: Calculate distance from coastline
# ---------------------------------------------------------
print("Calculating distance from coastline...")

distance_to_coast = distance_transform_edt(
    land_mask.astype(bool),
    sampling=(TARGET_RESOLUTION, TARGET_RESOLUTION),
).astype(np.float32)

distance_to_coast[valid_land_mask == 0] = 0.0


# ---------------------------------------------------------
# Step 9: Remove invalid areas from elevation
# ---------------------------------------------------------
elevation_land = elevation.copy()
elevation_land[valid_land_mask == 0] = np.nan


# ---------------------------------------------------------
# Step 10: Save processed rasters
# ---------------------------------------------------------
print("\nSaving aligned rasters...")

save_raster(
    "elevation_5m.tif",
    elevation_land,
    grid_transform,
    source_crs,
    "float32",
    -9999.0,
)

save_raster(
    "slope_degrees_5m.tif",
    slope_degrees,
    grid_transform,
    source_crs,
    "float32",
    -9999.0,
)

save_raster(
    "land_cover_class_5m.tif",
    land_cover_raster,
    grid_transform,
    source_crs,
    "uint16",
    0,
)

save_raster(
    "distance_to_coast_5m.tif",
    distance_to_coast,
    grid_transform,
    source_crs,
    "float32",
    -9999.0,
)

save_raster(
    "flood_target_5m.tif",
    flood_target,
    grid_transform,
    source_crs,
    "uint8",
    255,
)

save_raster(
    "valid_land_mask_5m.tif",
    valid_land_mask,
    grid_transform,
    source_crs,
    "uint8",
    0,
)


# ---------------------------------------------------------
# Step 11: Calculate statistics
# ---------------------------------------------------------
valid_cells = int(valid_land_mask.sum())
flood_cells = int(
    ((flood_target == 1) & (valid_land_mask == 1)).sum()
)
non_flood_cells = valid_cells - flood_cells

if valid_cells > 0:
    flood_percentage = (flood_cells / valid_cells) * 100
else:
    flood_percentage = 0.0

metadata = {
    "crs": str(source_crs),
    "resolution_metres": TARGET_RESOLUTION,
    "grid_width": grid_width,
    "grid_height": grid_height,
    "valid_land_cells": valid_cells,
    "flood_cells": flood_cells,
    "non_flood_cells": non_flood_cells,
    "flood_percentage": flood_percentage,
    "input_channels": [
        "elevation",
        "slope_degrees",
        "land_cover_class",
        "distance_to_coast",
    ],
    "target": "binary coastal flood susceptibility",
    "target_meaning": {
        "0": "non-flood-susceptible land",
        "1": "flood-susceptible land",
        "ignored": "water or invalid DEM area",
    },
}

metadata_path = OUTPUT_FOLDER / "raster_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print(f"SAVED: {metadata_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------
# Step 12: Create evidence figure
# ---------------------------------------------------------
print("Creating preprocessing evidence figure...")

figure, axes = plt.subplots(2, 3, figsize=(16, 10))

plots = [
    (elevation_land, "Elevation", "terrain"),
    (slope_degrees, "Slope (degrees)", "magma"),
    (land_cover_raster, "Land-cover class", "tab20"),
    (distance_to_coast, "Distance to coastline", "viridis"),
    (flood_target, "Binary flood target", "Reds"),
    (valid_land_mask, "Valid land mask", "Greys"),
]

for axis, (data, title, colour_map) in zip(axes.flat, plots):
    display_data = np.ma.masked_invalid(data)

    if title == "Land-cover class":
        display_data = np.ma.masked_where(
            valid_land_mask == 0,
            display_data,
        )

    if title == "Binary flood target":
        display_data = np.ma.masked_where(
            valid_land_mask == 0,
            display_data,
        )

    image = axis.imshow(display_data, cmap=colour_map)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

figure.suptitle(
    "ClimateAdapt AI – Aligned 5 m Model Rasters",
    fontsize=16,
    fontweight="bold",
)

figure.tight_layout(rect=[0, 0, 1, 0.96])

figure_path = EVIDENCE_FOLDER / "aligned_rasters_overview.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"SAVED: {figure_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------
print("\n" + "=" * 65)
print("PREPROCESSING COMPLETED")
print("=" * 65)
print(f"Valid land cells:     {valid_cells:,}")
print(f"Flood cells:          {flood_cells:,}")
print(f"Non-flood cells:      {non_flood_cells:,}")
print(f"Flood percentage:     {flood_percentage:.2f}%")
print(f"Output resolution:    {TARGET_RESOLUTION} metres")
print(f"Input channels:       4")
print("=" * 65)