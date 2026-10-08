from pathlib import Path
import json
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, distance_transform_edt


# =========================================================
# Configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_FOLDER = PROJECT_ROOT / "data/raw/dem"
LAND_COVER_FOLDER = PROJECT_ROOT / "data/raw/land_cover"
COASTLINE_FOLDER = PROJECT_ROOT / "data/raw/coastline"

LABEL_PATH = (
    PROJECT_ROOT
    / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_connected.geojson"
)

OUTPUT_FOLDER = PROJECT_ROOT / "data/processed/combined_rasters"
EVIDENCE_FOLDER = PROJECT_ROOT / "evidence/figures"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
EVIDENCE_FOLDER.mkdir(parents=True, exist_ok=True)

TARGET_RESOLUTION = 5.0
SPLIT_GAP_METRES = 100.0
AREA_CONTEXT_BUFFER_METRES = 750.0

SELECTED_AREAS = [
    "Tahunanui",
    "Rocks Road",
    "Nelson CBD",
    "Nelson Haven",
    "Waimea inlet",
]


# =========================================================
# Helper: save one GeoTIFF
# =========================================================
def save_raster(
    filename,
    array,
    transform,
    crs,
    dtype,
    nodata,
):
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


# =========================================================
# Step 1: Discover all source files
# =========================================================
dem_files = sorted(DEM_FOLDER.glob("*.tif"))
land_cover_files = sorted(LAND_COVER_FOLDER.glob("*.gpkg"))
coastline_files = sorted(COASTLINE_FOLDER.glob("*.gpkg"))

if not dem_files:
    raise FileNotFoundError("No DEM GeoTIFF files were found.")

if not land_cover_files:
    raise FileNotFoundError("No land-cover GeoPackages were found.")

if not coastline_files:
    raise FileNotFoundError("No coastline GeoPackages were found.")

print("\nSource files")
print("-" * 65)

for path in dem_files:
    print(f"DEM:        {path.name}")

for path in land_cover_files:
    print(f"Land cover: {path.name}")

for path in coastline_files:
    print(f"Coastline:  {path.name}")


# =========================================================
# Step 2: Inspect DEMs and create common 5 m grid
# =========================================================
print("\nCreating combined 5 m grid...")

minimum_x = float("inf")
minimum_y = float("inf")
maximum_x = float("-inf")
maximum_y = float("-inf")

dem_crs = None

for dem_file in dem_files:
    with rasterio.open(dem_file) as source:
        if dem_crs is None:
            dem_crs = source.crs
        elif source.crs != dem_crs:
            raise ValueError(
                f"CRS mismatch in {dem_file.name}: "
                f"{source.crs} does not match {dem_crs}"
            )

        minimum_x = min(minimum_x, source.bounds.left)
        minimum_y = min(minimum_y, source.bounds.bottom)
        maximum_x = max(maximum_x, source.bounds.right)
        maximum_y = max(maximum_y, source.bounds.top)

grid_width = math.ceil(
    (maximum_x - minimum_x) / TARGET_RESOLUTION
)

grid_height = math.ceil(
    (maximum_y - minimum_y) / TARGET_RESOLUTION
)

grid_transform = from_origin(
    minimum_x,
    maximum_y,
    TARGET_RESOLUTION,
    TARGET_RESOLUTION,
)

print(f"Combined CRS: {dem_crs}")
print(f"Grid width: {grid_width}")
print(f"Grid height: {grid_height}")
print(f"Resolution: {TARGET_RESOLUTION} metres")


# =========================================================
# Step 3: Resample and combine DEMs
# =========================================================
print("\nCombining DEM files...")

elevation_sum = np.zeros(
    (grid_height, grid_width),
    dtype=np.float32,
)

elevation_count = np.zeros(
    (grid_height, grid_width),
    dtype=np.uint8,
)

for dem_file in dem_files:
    print(f"Processing DEM: {dem_file.name}")

    temporary_dem = np.full(
        (grid_height, grid_width),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(dem_file) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=temporary_dem,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=grid_transform,
            dst_crs=dem_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

    valid_temporary = np.isfinite(temporary_dem)

    elevation_sum[valid_temporary] += temporary_dem[
        valid_temporary
    ]

    elevation_count[valid_temporary] += 1

elevation = np.full(
    (grid_height, grid_width),
    np.nan,
    dtype=np.float32,
)

valid_elevation = elevation_count > 0

elevation[valid_elevation] = (
    elevation_sum[valid_elevation]
    / elevation_count[valid_elevation]
)


# =========================================================
# Step 4: Load and combine vector datasets
# =========================================================
print("\nCombining land-cover files...")

land_cover_parts = []

for path in land_cover_files:
    data = gpd.read_file(path).to_crs(dem_crs)
    land_cover_parts.append(data)

land_cover = gpd.GeoDataFrame(
    pd.concat(
        land_cover_parts,
        ignore_index=True,
    ),
    crs=dem_crs,
)

print(f"Combined land-cover features: {len(land_cover)}")


print("\nCombining coastline files...")

coastline_parts = []

for path in coastline_files:
    data = gpd.read_file(path).to_crs(dem_crs)
    coastline_parts.append(data)

coastline = gpd.GeoDataFrame(
    pd.concat(
        coastline_parts,
        ignore_index=True,
    ),
    crs=dem_crs,
)

print(f"Combined coastline features: {len(coastline)}")


# =========================================================
# Step 5: Load selected NCC flood labels
# =========================================================
print("\nLoading NCC flood labels...")

all_labels = gpd.read_file(LABEL_PATH).to_crs(dem_crs)

labels = all_labels[
    all_labels["InundationCellName"].isin(SELECTED_AREAS)
].copy()

available_areas = sorted(
    labels["InundationCellName"].unique().tolist()
)

print("Selected coastal areas found:")

for area in available_areas:
    print(f"  - {area}")

missing_areas = sorted(
    set(SELECTED_AREAS) - set(available_areas)
)

if missing_areas:
    raise ValueError(
        f"Missing selected flood-label areas: {missing_areas}"
    )

area_labels = labels.dissolve(
    by="InundationCellName",
    as_index=False,
)


# =========================================================
# Step 6: Rasterize combined coastline as land mask
# =========================================================
print("\nCreating combined land mask...")

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


# =========================================================
# Step 7: Rasterize combined land cover
# =========================================================
print("Creating combined land-cover raster...")

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
            (
                geometry,
                int(float(class_value)),
            )
        )

land_cover_raster = rasterize(
    land_cover_shapes,
    out_shape=(grid_height, grid_width),
    transform=grid_transform,
    fill=0,
    all_touched=True,
    dtype=np.uint16,
)

land_cover_raster[land_mask == 0] = 0


# =========================================================
# Step 8: Create combined flood target
# =========================================================
print("Creating combined binary flood target...")

flood_shapes = [
    (geometry, 1)
    for geometry in labels.geometry
    if geometry is not None and not geometry.is_empty
]

raw_flood_target = rasterize(
    flood_shapes,
    out_shape=(grid_height, grid_width),
    transform=grid_transform,
    fill=0,
    all_touched=True,
    dtype=np.uint8,
)

valid_land_mask = (
    (land_mask == 1)
    & np.isfinite(elevation)
).astype(np.uint8)

flood_target = (
    (raw_flood_target == 1)
    & (valid_land_mask == 1)
).astype(np.uint8)


# =========================================================
# Step 9: Calculate slope
# =========================================================
print("Calculating combined slope...")

finite_elevation = elevation[np.isfinite(elevation)]

if len(finite_elevation) == 0:
    raise ValueError("Combined DEM contains no valid values.")

replacement_elevation = float(
    np.median(finite_elevation)
)

elevation_for_slope = np.where(
    np.isfinite(elevation),
    elevation,
    replacement_elevation,
)

gradient_y, gradient_x = np.gradient(
    elevation_for_slope,
    TARGET_RESOLUTION,
    TARGET_RESOLUTION,
)

slope_degrees = np.degrees(
    np.arctan(
        np.sqrt(
            (gradient_x ** 2)
            + (gradient_y ** 2)
        )
    )
).astype(np.float32)

slope_degrees[valid_land_mask == 0] = np.nan


# =========================================================
# Step 10: Calculate distance to coastline
# =========================================================
print("Calculating distance to coastline...")

distance_to_coast = distance_transform_edt(
    land_mask.astype(bool),
    sampling=(
        TARGET_RESOLUTION,
        TARGET_RESOLUTION,
    ),
).astype(np.float32)

distance_to_coast[valid_land_mask == 0] = np.nan


# =========================================================
# Step 11: Create named geographic model zones
# =========================================================
print("Creating geographic data split...")

area_geometries = {
    row["InundationCellName"]: row.geometry
    for _, row in area_labels.iterrows()
}

training_geometry = (
    area_geometries["Nelson CBD"]
    .union(area_geometries["Nelson Haven"])
    .buffer(AREA_CONTEXT_BUFFER_METRES)
)

validation_geometry = (
    area_geometries["Rocks Road"]
    .buffer(AREA_CONTEXT_BUFFER_METRES)
)

testing_geometry = (
    area_geometries["Tahunanui"]
    .buffer(AREA_CONTEXT_BUFFER_METRES)
)

demonstration_geometry = (
    area_geometries["Waimea inlet"]
    .buffer(AREA_CONTEXT_BUFFER_METRES)
)


def rasterize_zone(geometry):
    return rasterize(
        [(geometry, 1)],
        out_shape=(grid_height, grid_width),
        transform=grid_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    ).astype(bool)


raw_zone_masks = {
    1: rasterize_zone(training_geometry),
    2: rasterize_zone(validation_geometry),
    3: rasterize_zone(testing_geometry),
    4: rasterize_zone(demonstration_geometry),
}

# Exact official label polygons for each named zone. These positive cores must
# survive the separation operation below; only overlapping buffer/context
# cells should be discarded.
core_zone_masks = {
    1: rasterize_zone(
        area_geometries["Nelson CBD"].union(
            area_geometries["Nelson Haven"]
        )
    ),
    2: rasterize_zone(area_geometries["Rocks Road"]),
    3: rasterize_zone(area_geometries["Tahunanui"]),
    4: rasterize_zone(area_geometries["Waimea inlet"]),
}

# Remove cells close to a different split. Each side gives up half
# the requested gap, producing approximately 100 m separation.
half_gap_pixels = max(
    1,
    math.ceil(
        (SPLIT_GAP_METRES / 2)
        / TARGET_RESOLUTION
    ),
)

safe_zone_masks = {}

for zone_id, zone_mask in raw_zone_masks.items():
    other_zone = np.zeros_like(
        zone_mask,
        dtype=bool,
    )

    for other_id, other_mask in raw_zone_masks.items():
        if other_id != zone_id:
            other_zone |= other_mask

    expanded_other_zone = binary_dilation(
        other_zone,
        iterations=half_gap_pixels,
    )

    separated_context = zone_mask & ~expanded_other_zone

    safe_zone_masks[zone_id] = (
        (separated_context | core_zone_masks[zone_id])
        & (valid_land_mask == 1)
    )

split_mask = np.zeros(
    (grid_height, grid_width),
    dtype=np.uint8,
)

for zone_id, zone_mask in safe_zone_masks.items():
    split_mask[zone_mask] = zone_id

print(
    "Named zones: training=Nelson CBD/Haven, "
    "validation=Rocks Road, testing=Tahunanui, "
    "demonstration=Waimea inlet/Monaco"
)


# =========================================================
# Step 12: Create label-area ID raster
# =========================================================
area_id_mapping = {
    "Nelson Haven": 1,
    "Nelson CBD": 2,
    "Rocks Road": 3,
    "Tahunanui": 4,
    "Waimea inlet": 5,
}

area_id_shapes = []

for _, row in area_labels.iterrows():
    name = row["InundationCellName"]

    area_id_shapes.append(
        (
            row.geometry,
            area_id_mapping[name],
        )
    )

flood_area_id = rasterize(
    area_id_shapes,
    out_shape=(grid_height, grid_width),
    transform=grid_transform,
    fill=0,
    all_touched=True,
    dtype=np.uint8,
)

flood_area_id[valid_land_mask == 0] = 0


# =========================================================
# Step 13: Prepare arrays for saving
# =========================================================
elevation_land = elevation.copy()
elevation_land[valid_land_mask == 0] = np.nan

flood_target_output = np.where(
    valid_land_mask == 1,
    flood_target,
    255,
).astype(np.uint8)


# =========================================================
# Step 14: Save combined rasters
# =========================================================
print("\nSaving combined rasters...")

save_raster(
    "elevation_5m.tif",
    elevation_land,
    grid_transform,
    dem_crs,
    "float32",
    -9999.0,
)

save_raster(
    "slope_degrees_5m.tif",
    slope_degrees,
    grid_transform,
    dem_crs,
    "float32",
    -9999.0,
)

save_raster(
    "land_cover_class_5m.tif",
    land_cover_raster,
    grid_transform,
    dem_crs,
    "uint16",
    0,
)

save_raster(
    "distance_to_coast_5m.tif",
    distance_to_coast,
    grid_transform,
    dem_crs,
    "float32",
    -9999.0,
)

save_raster(
    "flood_target_5m.tif",
    flood_target_output,
    grid_transform,
    dem_crs,
    "uint8",
    255,
)

save_raster(
    "valid_land_mask_5m.tif",
    valid_land_mask,
    grid_transform,
    dem_crs,
    "uint8",
    0,
)

save_raster(
    "split_mask_5m.tif",
    split_mask,
    grid_transform,
    dem_crs,
    "uint8",
    0,
)

save_raster(
    "flood_area_id_5m.tif",
    flood_area_id,
    grid_transform,
    dem_crs,
    "uint8",
    0,
)


# =========================================================
# Step 15: Calculate split statistics
# =========================================================
split_names = {
    1: "training",
    2: "validation",
    3: "testing",
    4: "demonstration",
}

split_statistics = {}

print("\n" + "=" * 72)
print("GEOGRAPHIC SPLIT STATISTICS")
print("=" * 72)

for split_id, split_name in split_names.items():
    split_cells = split_mask == split_id

    valid_cells = int(split_cells.sum())
    flood_cells = int(
        (
            split_cells
            & (flood_target == 1)
        ).sum()
    )

    non_flood_cells = valid_cells - flood_cells

    flood_percentage = (
        (flood_cells / valid_cells) * 100
        if valid_cells > 0
        else 0.0
    )

    split_statistics[split_name] = {
        "valid_cells": valid_cells,
        "flood_cells": flood_cells,
        "non_flood_cells": non_flood_cells,
        "flood_percentage": flood_percentage,
    }

    print(
        f"{split_name.capitalize():<12} "
        f"cells={valid_cells:>9,}  "
        f"flood={flood_cells:>8,}  "
        f"non-flood={non_flood_cells:>9,}  "
        f"flood%={flood_percentage:>6.2f}"
    )

print("=" * 72)


# =========================================================
# Step 16: Save metadata
# =========================================================
metadata = {
    "crs": str(dem_crs),
    "resolution_metres": TARGET_RESOLUTION,
    "grid_width": grid_width,
    "grid_height": grid_height,
    "dem_files": [
        path.name for path in dem_files
    ],
    "land_cover_files": [
        path.name for path in land_cover_files
    ],
    "coastline_files": [
        path.name for path in coastline_files
    ],
    "selected_flood_areas": SELECTED_AREAS,
    "input_channels": [
        "elevation",
        "slope_degrees",
        "land_cover_class",
        "distance_to_coast",
    ],
    "split_meaning": {
        "0": "ignored area or separation gap",
        "1": "training: Nelson CBD and Nelson Haven",
        "2": "validation: Rocks Road",
        "3": "testing: Tahunanui",
        "4": "demonstration holdout: Waimea inlet and Monaco",
    },
    "split_gap_metres": SPLIT_GAP_METRES,
    "area_context_buffer_metres": AREA_CONTEXT_BUFFER_METRES,
    "split_statistics": split_statistics,
}

metadata_path = OUTPUT_FOLDER / "combined_metadata.json"

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )

print(f"SAVED: {metadata_path.relative_to(PROJECT_ROOT)}")


# =========================================================
# Step 17: Create evidence figure
# =========================================================
print("Creating combined-raster evidence figure...")

figure, axes = plt.subplots(
    2,
    4,
    figsize=(20, 11),
)

elevation_display = np.ma.masked_invalid(
    elevation_land
)

slope_display = np.ma.masked_invalid(
    slope_degrees
)

land_cover_display = np.ma.masked_where(
    valid_land_mask == 0,
    land_cover_raster,
)

distance_display = np.ma.masked_invalid(
    distance_to_coast
)

target_display = np.ma.masked_where(
    valid_land_mask == 0,
    flood_target,
)

valid_display = np.ma.masked_where(
    valid_land_mask == 0,
    valid_land_mask,
)

area_display = np.ma.masked_where(
    flood_area_id == 0,
    flood_area_id,
)

split_display = np.ma.masked_where(
    split_mask == 0,
    split_mask,
)

plot_items = [
    (
        elevation_display,
        "Elevation",
        "terrain",
    ),
    (
        slope_display,
        "Slope (degrees)",
        "magma",
    ),
    (
        land_cover_display,
        "Land-cover class",
        "tab20",
    ),
    (
        distance_display,
        "Distance to coastline",
        "viridis",
    ),
    (
        target_display,
        "Binary flood target",
        "Reds",
    ),
    (
        valid_display,
        "Valid land mask",
        "Greys",
    ),
    (
        area_display,
        "Flood-area ID",
        "tab10",
    ),
]

for axis, (
    display_data,
    title,
    colour_map,
) in zip(
    axes.flat[:7],
    plot_items,
):
    image = axis.imshow(
        display_data,
        cmap=colour_map,
    )

    axis.set_title(title)
    axis.axis("off")

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )


split_colour_map = ListedColormap(
    [
        "#2ca02c",
        "#ffbf00",
        "#d62728",
        "#1f77b4",
    ]
)

axes.flat[7].imshow(
    split_display,
    cmap=split_colour_map,
    vmin=1,
    vmax=4,
)

axes.flat[7].set_title(
    "Geographic model split"
)

axes.flat[7].axis("off")

split_legend = [
    Patch(
        facecolor="#2ca02c",
        label="Training",
    ),
    Patch(
        facecolor="#ffbf00",
        label="Validation",
    ),
    Patch(
        facecolor="#d62728",
        label="Testing",
    ),
    Patch(
        facecolor="#1f77b4",
        label="Waimea/Monaco demonstration",
    ),
    Patch(
        facecolor="white",
        edgecolor="black",
        label="Ignored/gap",
    ),
]

axes.flat[7].legend(
    handles=split_legend,
    loc="lower right",
)

figure.suptitle(
    "ClimateAdapt AI – Combined 5 m Model Dataset",
    fontsize=18,
    fontweight="bold",
)

figure.tight_layout(
    rect=[0, 0, 1, 0.96]
)

figure_path = (
    EVIDENCE_FOLDER
    / "combined_rasters_and_split.png"
)

plt.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print(f"SAVED: {figure_path.relative_to(PROJECT_ROOT)}")


# =========================================================
# Final result
# =========================================================
print("\n" + "=" * 72)
print("COMBINED PREPROCESSING COMPLETED")
print("=" * 72)
print(f"Grid: {grid_width} x {grid_height}")
print(f"Resolution: {TARGET_RESOLUTION} metres")
print("Training region: Nelson CBD + Nelson Haven")
print("Validation region: Rocks Road")
print("Testing region: Tahunanui")
print("Demonstration region: Waimea inlet + Monaco")
print(f"Area context buffer: {AREA_CONTEXT_BUFFER_METRES} metres")
print(f"Separation gap: {SPLIT_GAP_METRES} metres")
print("=" * 72)
