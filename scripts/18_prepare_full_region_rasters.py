from __future__ import annotations

import gc
import json
import math
import shutil
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEM_ROOT = PROJECT_ROOT / "data" / "raw" / "nelson_region_full" / "dem"
VECTOR_ROOT = PROJECT_ROOT / "data" / "processed" / "full_region" / "vectors"

LAND_COVER_PATH = VECTOR_ROOT / "lcdb_nelson_combined_valid.gpkg"
COASTLINE_PATH = VECTOR_ROOT / "coastline_nelson_valid.gpkg"
LABEL_PATH = VECTOR_ROOT / "ncc_coastal_labels_valid.gpkg"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "full_region" / "rasters"
ELEVATION_PATH = OUTPUT_ROOT / "elevation_5m.tif"
SLOPE_PATH = OUTPUT_ROOT / "slope_degrees_5m.tif"
LAND_COVER_RASTER_PATH = OUTPUT_ROOT / "land_cover_class_5m.tif"
DISTANCE_PATH = OUTPUT_ROOT / "distance_to_coast_5m.tif"
VALID_MASK_PATH = OUTPUT_ROOT / "valid_land_mask_5m.tif"
FLOOD_TARGET_PATH = OUTPUT_ROOT / "flood_target_5m.tif"
FLOOD_AREA_ID_PATH = OUTPUT_ROOT / "flood_area_id_5m.tif"
METADATA_PATH = OUTPUT_ROOT / "full_region_metadata.json"

FIGURE_PATH = PROJECT_ROOT / "evidence" / "figures" / "full_region_rasters_overview.png"

CRS = "EPSG:2193"
RESOLUTION = 5.0
FLOAT_NODATA = -9999.0
BLOCK_SIZE = 512
MINIMUM_FREE_GB = 3.0


def first_layer(path: Path) -> str:
    layers = gpd.list_layers(path)
    if layers.empty:
        raise ValueError(f"No readable layer found in {path}")
    return str(layers.iloc[0]["name"])


def read_vector(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = gpd.read_file(path, layer=first_layer(path))
    if data.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    data = data.to_crs(CRS)
    invalid = int((~data.geometry.is_valid).sum())
    if invalid:
        raise ValueError(f"{path} still contains {invalid} invalid geometries")
    return data


def find_column(data: gpd.GeoDataFrame, candidates: list[str]) -> str:
    lookup = {column.casefold(): column for column in data.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    raise KeyError(
        f"Could not find any of {candidates}. Available columns: {list(data.columns)}"
    )


def aligned_grid(dem_paths: list[Path]) -> tuple[object, int, int, dict]:
    bounds = []
    for path in dem_paths:
        with rasterio.open(path) as source:
            if str(source.crs) != CRS:
                raise ValueError(f"Unexpected CRS in {path}: {source.crs}")
            bounds.append(source.bounds)

    left = math.floor(min(item.left for item in bounds) / RESOLUTION) * RESOLUTION
    bottom = math.floor(min(item.bottom for item in bounds) / RESOLUTION) * RESOLUTION
    right = math.ceil(max(item.right for item in bounds) / RESOLUTION) * RESOLUTION
    top = math.ceil(max(item.top for item in bounds) / RESOLUTION) * RESOLUTION
    width = int(round((right - left) / RESOLUTION))
    height = int(round((top - bottom) / RESOLUTION))
    transform = from_origin(left, top, RESOLUTION, RESOLUTION)
    grid_bounds = {"left": left, "bottom": bottom, "right": right, "top": top}
    return transform, width, height, grid_bounds


def profile_for(
    width: int,
    height: int,
    transform,
    dtype: str,
    nodata,
    predictor: int | None = None,
) -> dict:
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "crs": CRS,
        "transform": transform,
        "dtype": dtype,
        "nodata": nodata,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "BIGTIFF": "YES",
        "interleave": "band",
    }
    if predictor is not None:
        profile["predictor"] = predictor
    return profile


def initialise_raster(path: Path, profile: dict, fill_value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        for _, window in destination.block_windows(1):
            block = np.full(
                (int(window.height), int(window.width)),
                fill_value,
                dtype=profile["dtype"],
            )
            destination.write(block, 1, window=window)


def create_elevation(
    dem_paths: list[Path], transform, width: int, height: int
) -> None:
    profile = profile_for(
        width, height, transform, "float32", FLOAT_NODATA, predictor=3
    )
    initialise_raster(ELEVATION_PATH, profile, FLOAT_NODATA)

    with rasterio.open(ELEVATION_PATH, "r+") as destination:
        for index, path in enumerate(dem_paths, start=1):
            print(f"  Reprojecting DEM tile {index}/{len(dem_paths)}: {path.name}")
            with rasterio.open(path) as source:
                reproject(
                    source=rasterio.band(source, 1),
                    destination=rasterio.band(destination, 1),
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=transform,
                    dst_crs=CRS,
                    dst_nodata=FLOAT_NODATA,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=False,
                    num_threads=2,
                    warp_mem_limit=256,
                )


def rasterize_array(
    shapes,
    height: int,
    width: int,
    transform,
    dtype,
    fill=0,
) -> np.ndarray:
    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill,
        dtype=dtype,
        all_touched=False,
    )


def save_array(path: Path, array: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        for _, window in destination.block_windows(1):
            row_start = int(window.row_off)
            row_end = row_start + int(window.height)
            col_start = int(window.col_off)
            col_end = col_start + int(window.width)
            destination.write(
                array[row_start:row_end, col_start:col_end], 1, window=window
            )


def create_valid_mask(
    land_mask: np.ndarray, transform, width: int, height: int
) -> tuple[int, int]:
    profile = profile_for(width, height, transform, "uint8", 0, predictor=2)
    valid_count = 0
    missing_dem_land_count = 0

    with rasterio.open(ELEVATION_PATH) as elevation, rasterio.open(
        VALID_MASK_PATH, "w", **profile
    ) as destination:
        for _, window in destination.block_windows(1):
            row_start = int(window.row_off)
            row_end = row_start + int(window.height)
            col_start = int(window.col_off)
            col_end = col_start + int(window.width)

            elevation_block = elevation.read(1, window=window)
            land_block = land_mask[row_start:row_end, col_start:col_end] > 0
            elevation_valid = (
                np.isfinite(elevation_block) & (elevation_block != FLOAT_NODATA)
            )
            valid = land_block & elevation_valid
            missing_dem_land_count += int(np.count_nonzero(land_block & ~elevation_valid))
            valid_count += int(np.count_nonzero(valid))
            destination.write(valid.astype(np.uint8), 1, window=window)

    return valid_count, missing_dem_land_count


def expanded_window(window: Window, width: int, height: int, halo: int = 1) -> Window:
    col_start = max(0, int(window.col_off) - halo)
    row_start = max(0, int(window.row_off) - halo)
    col_end = min(width, int(window.col_off + window.width) + halo)
    row_end = min(height, int(window.row_off + window.height) + halo)
    return Window(col_start, row_start, col_end - col_start, row_end - row_start)


def create_slope(transform, width: int, height: int) -> None:
    profile = profile_for(
        width, height, transform, "float32", FLOAT_NODATA, predictor=3
    )
    with rasterio.open(ELEVATION_PATH) as elevation, rasterio.open(
        SLOPE_PATH, "w", **profile
    ) as destination:
        total_blocks = sum(1 for _ in destination.block_windows(1))
        for block_index, (_, window) in enumerate(destination.block_windows(1), start=1):
            expanded = expanded_window(window, width, height, halo=1)
            data = elevation.read(1, window=expanded).astype(np.float32)
            valid = np.isfinite(data) & (data != FLOAT_NODATA)
            data[~valid] = np.nan

            gradient_y, gradient_x = np.gradient(data, RESOLUTION, RESOLUTION)
            slope = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y))).astype(
                np.float32
            )
            slope[~np.isfinite(slope)] = FLOAT_NODATA

            row_offset = int(window.row_off - expanded.row_off)
            col_offset = int(window.col_off - expanded.col_off)
            core = slope[
                row_offset : row_offset + int(window.height),
                col_offset : col_offset + int(window.width),
            ]
            destination.write(core, 1, window=window)

            if block_index % 50 == 0 or block_index == total_blocks:
                print(f"  Slope blocks processed: {block_index}/{total_blocks}")


def create_distance(
    land_mask: np.ndarray,
    transform,
    width: int,
    height: int,
) -> None:
    print("  Running Euclidean distance transform...")
    distance = distance_transform_edt(land_mask > 0, sampling=RESOLUTION)
    distance = distance.astype(np.float32)

    with rasterio.open(VALID_MASK_PATH) as valid_source:
        for _, window in valid_source.block_windows(1):
            row_start = int(window.row_off)
            row_end = row_start + int(window.height)
            col_start = int(window.col_off)
            col_end = col_start + int(window.width)
            valid = valid_source.read(1, window=window) > 0
            block = distance[row_start:row_end, col_start:col_end]
            block[~valid] = FLOAT_NODATA

    profile = profile_for(
        width, height, transform, "float32", FLOAT_NODATA, predictor=3
    )
    save_array(DISTANCE_PATH, distance, profile)
    del distance
    gc.collect()


def preview_array(path: Path, max_dimension: int = 1400) -> np.ndarray:
    with rasterio.open(path) as source:
        scale = max(source.width / max_dimension, source.height / max_dimension, 1.0)
        output_width = max(1, int(round(source.width / scale)))
        output_height = max(1, int(round(source.height / scale)))
        return source.read(
            1,
            out_shape=(output_height, output_width),
            resampling=Resampling.nearest,
            masked=True,
        )


def create_figure() -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    layers = [
        (ELEVATION_PATH, "Elevation", "terrain"),
        (SLOPE_PATH, "Slope (degrees)", "magma"),
        (LAND_COVER_RASTER_PATH, "Land-cover class", "tab20"),
        (DISTANCE_PATH, "Distance to coastline", "viridis"),
        (FLOOD_TARGET_PATH, "Flood target", "Reds"),
        (FLOOD_AREA_ID_PATH, "Flood-area ID", "tab10"),
        (VALID_MASK_PATH, "Valid land mask", "gray_r"),
    ]

    figure, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    for axis, (path, title, colour_map) in zip(axes, layers):
        image = axis.imshow(preview_array(path), cmap=colour_map)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axes[-1].set_axis_off()
    figure.suptitle("ClimateAdapt AI – Full Nelson 5 m Model Rasters", fontsize=18)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    start_time = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    disk = shutil.disk_usage(PROJECT_ROOT)
    free_gb = disk.free / (1024**3)
    if free_gb < MINIMUM_FREE_GB:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB free. At least {MINIMUM_FREE_GB:.1f} GB is required."
        )

    dem_paths = sorted(DEM_ROOT.rglob("*.tif"))
    if not dem_paths:
        raise FileNotFoundError(f"No DEM GeoTIFFs found under {DEM_ROOT}")

    land_cover = read_vector(LAND_COVER_PATH)
    coastline = read_vector(COASTLINE_PATH)
    labels = read_vector(LABEL_PATH)
    transform, width, height, grid_bounds = aligned_grid(dem_paths)
    cell_count = width * height

    print("\n" + "=" * 80)
    print("CLIMATEADAPT AI – FULL-REGION 5 M RASTER PREPARATION")
    print("=" * 80)
    print(f"DEM tiles:          {len(dem_paths)}")
    print(f"Grid:               {width:,} x {height:,}")
    print(f"Grid cells:         {cell_count:,}")
    print(f"Resolution:         {RESOLUTION:.1f} metres")
    print(f"Free disk space:    {free_gb:.2f} GB")
    print(f"Land-cover records: {len(land_cover):,}")
    print(f"Coastline records:  {len(coastline):,}")
    print(f"Flood-label areas:  {len(labels):,}")

    print("\n[1/7] Creating the 5 m elevation mosaic...")
    create_elevation(dem_paths, transform, width, height)

    print("\n[2/7] Rasterising the coastline-derived land mask...")
    land_mask = rasterize_array(
        ((geometry, 1) for geometry in coastline.geometry),
        height,
        width,
        transform,
        "uint8",
    )
    valid_count, missing_dem_land_count = create_valid_mask(
        land_mask, transform, width, height
    )

    print("\n[3/7] Rasterising LCDB 2023 land-cover classes...")
    class_column = find_column(land_cover, ["Class_2023", "class_2023"])
    land_cover_shapes = (
        (geometry, int(value))
        for geometry, value in zip(land_cover.geometry, land_cover[class_column])
        if geometry is not None
        and not geometry.is_empty
        and pd.notna(value)
    )
    land_cover_array = rasterize_array(
        land_cover_shapes, height, width, transform, "uint16"
    )
    land_cover_profile = profile_for(
        width, height, transform, "uint16", 0, predictor=2
    )
    save_array(LAND_COVER_RASTER_PATH, land_cover_array, land_cover_profile)
    land_cover_classes = sorted(int(value) for value in np.unique(land_cover_array))
    del land_cover_array
    gc.collect()

    print("\n[4/7] Rasterising repaired flood targets and area IDs...")
    area_column = find_column(labels, ["InundationCellName", "name"])
    area_names = sorted(str(value) for value in labels[area_column].unique())
    area_id_mapping = {name: index + 1 for index, name in enumerate(area_names)}
    flood_target = rasterize_array(
        ((geometry, 1) for geometry in labels.geometry),
        height,
        width,
        transform,
        "uint8",
    )
    area_ids = rasterize_array(
        (
            (row.geometry, area_id_mapping[str(row[area_column])])
            for _, row in labels.iterrows()
        ),
        height,
        width,
        transform,
        "uint8",
    )
    binary_profile = profile_for(width, height, transform, "uint8", 0, predictor=2)
    save_array(FLOOD_TARGET_PATH, flood_target, binary_profile)
    save_array(FLOOD_AREA_ID_PATH, area_ids, binary_profile)

    with rasterio.open(VALID_MASK_PATH) as valid_source:
        valid_mask = valid_source.read(1) > 0
    valid_flood_count = int(np.count_nonzero((flood_target > 0) & valid_mask))
    outside_valid_flood_count = int(np.count_nonzero((flood_target > 0) & ~valid_mask))
    del flood_target, area_ids, valid_mask
    gc.collect()

    print("\n[5/7] Calculating slope in small raster windows...")
    create_slope(transform, width, height)

    print("\n[6/7] Calculating distance to coastline...")
    create_distance(land_mask, transform, width, height)
    del land_mask
    gc.collect()

    print("\n[7/7] Writing metadata and evidence figure...")
    elapsed_seconds = time.perf_counter() - start_time
    metadata = {
        "crs": CRS,
        "resolution_metres": RESOLUTION,
        "width": width,
        "height": height,
        "cell_count": cell_count,
        "bounds": grid_bounds,
        "dem_tile_count": len(dem_paths),
        "dem_tiles": [str(path.relative_to(PROJECT_ROOT)) for path in dem_paths],
        "valid_land_cells": valid_count,
        "valid_land_area_km2": valid_count * RESOLUTION * RESOLUTION / 1_000_000,
        "land_cells_without_dem": missing_dem_land_count,
        "flood_cells_on_valid_land": valid_flood_count,
        "flood_cells_outside_valid_land": outside_valid_flood_count,
        "land_cover_class_column": class_column,
        "land_cover_classes": land_cover_classes,
        "flood_area_id_mapping": area_id_mapping,
        "processing_seconds": elapsed_seconds,
        "outputs": {
            "elevation": ELEVATION_PATH.name,
            "slope": SLOPE_PATH.name,
            "land_cover": LAND_COVER_RASTER_PATH.name,
            "distance_to_coast": DISTANCE_PATH.name,
            "valid_land_mask": VALID_MASK_PATH.name,
            "flood_target": FLOOD_TARGET_PATH.name,
            "flood_area_id": FLOOD_AREA_ID_PATH.name,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    create_figure()

    print("\n" + "=" * 80)
    print("FULL-REGION RASTER PREPARATION COMPLETED")
    print("=" * 80)
    print(f"Grid:                         {width:,} x {height:,}")
    print(f"Valid land cells:             {valid_count:,}")
    print(f"Valid land area:              {metadata['valid_land_area_km2']:.3f} km²")
    print(f"Land cells without DEM:       {missing_dem_land_count:,}")
    print(f"Flood cells on valid land:    {valid_flood_count:,}")
    print(f"Flood cells outside valid:    {outside_valid_flood_count:,}")
    print(f"Land-cover classes:           {land_cover_classes}")
    print(f"Processing time:              {elapsed_seconds / 60:.2f} minutes")
    print("-" * 80)
    print(f"Rasters:  {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure:   {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    print("No training, validation or test model artifacts were modified.")


if __name__ == "__main__":
    main()
