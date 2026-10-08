"""Run the frozen Experiment 2 U-Net across the full Nelson raster coverage.

This is label-blind operational screening inference. The script deliberately
does not open the flood-target or flood-area rasters, retrain the model, or
change the validation-selected threshold. Predictions are accumulated in
disk-backed arrays so the 75-million-cell grid does not need to fit in RAM.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.models.unet_v2 import CompactUNetV2  # noqa: E402


RASTER_DIR = PROJECT_ROOT / "data" / "processed" / "full_region" / "rasters"
TILE_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "tiles" / "tile_metadata.json"
)
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "climateadapt_unet_v2_best.pt"
THRESHOLD_PATH = (
    PROJECT_ROOT / "outputs" / "validation_v2" / "selected_threshold_v2.json"
)

ELEVATION_PATH = RASTER_DIR / "elevation_5m.tif"
SLOPE_PATH = RASTER_DIR / "slope_degrees_5m.tif"
DISTANCE_PATH = RASTER_DIR / "distance_to_coast_5m.tif"
LAND_COVER_PATH = RASTER_DIR / "land_cover_class_5m.tif"
VALID_MASK_PATH = RASTER_DIR / "valid_land_mask_5m.tif"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_region_inference_v2"
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures"
TEMP_DIR = OUTPUT_DIR / "_temporary_accumulation"

PROBABILITY_PATH = OUTPUT_DIR / "nelson_probability_v2.tif"
BINARY_PATH = OUTPUT_DIR / "nelson_susceptibility_v2.tif"
UNCERTAINTY_PATH = OUTPUT_DIR / "nelson_uncertainty_v2.tif"
COVERAGE_PATH = OUTPUT_DIR / "nelson_inference_coverage_v2.tif"
SUMMARY_JSON = OUTPUT_DIR / "nelson_full_region_summary_v2.json"
SUMMARY_CSV = OUTPUT_DIR / "nelson_full_region_summary_v2.csv"
EVIDENCE_FIGURE = FIGURE_DIR / "nelson_full_region_inference_v2.png"

FLOAT_NODATA = -9999.0
BYTE_NODATA = 255
DEFAULT_BATCH_SIZE = 4
MINIMUM_FREE_DISK_GB = 3.0


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length < tile_size:
        raise ValueError(
            f"Raster dimension {length} is smaller than tile size {tile_size}."
        )
    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def extract_state_dict(checkpoint: dict) -> dict:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "model_weights"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def clean_state_dict(state_dict: dict) -> dict:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def infer_dimensions(state_dict: dict) -> tuple[int, int]:
    convolution_weights = [
        value
        for value in state_dict.values()
        if torch.is_tensor(value) and value.ndim == 4
    ]
    if not convolution_weights:
        raise RuntimeError("No convolution weights were found in the checkpoint.")
    return (
        int(convolution_weights[0].shape[1]),
        int(convolution_weights[0].shape[0]),
    )


def validate_raster_alignment(paths: list[Path]) -> dict:
    reference = None
    for path in paths:
        with rasterio.open(path) as source:
            description = {
                "width": source.width,
                "height": source.height,
                "transform": source.transform,
                "crs": source.crs,
            }
        if reference is None:
            reference = description
        elif description != reference:
            raise ValueError(f"Raster alignment mismatch: {path}")
    if reference is None:
        raise RuntimeError("No raster inputs were supplied.")
    return reference


def normalise_feature(array: np.ndarray, statistics: dict) -> np.ndarray:
    working = array.astype(np.float32, copy=True)
    nodata = ~np.isfinite(working) | (working == FLOAT_NODATA)
    if bool(statistics.get("log1p_transformation", False)):
        working = np.log1p(np.maximum(working, 0.0))

    lower = float(statistics["lower_percentile"])
    upper = float(statistics["upper_percentile"])
    mean = float(statistics["mean"])
    standard_deviation = max(float(statistics["standard_deviation"]), 1e-6)

    working = np.clip(working, lower, upper)
    working = (working - mean) / standard_deviation
    working[nodata | ~np.isfinite(working)] = 0.0
    return working.astype(np.float32)


def blend_weights(tile_size: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, tile_size, dtype=np.float32)
    axis = 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * axis))
    return np.outer(axis, axis).astype(np.float32)


def output_profile(reference_profile: dict, dtype: str, nodata) -> dict:
    profile = reference_profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype == "float32" else 2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def create_memmap(path: Path, shape: tuple[int, int]) -> np.memmap:
    mapping = np.memmap(path, dtype="float32", mode="w+", shape=shape)
    mapping[:] = 0.0
    mapping.flush()
    return mapping


def read_features(
    window: Window,
    sources: dict,
    normalisation: dict,
    land_cover_classes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    valid = sources["valid"].read(1, window=window) == 1
    elevation = sources["elevation"].read(1, window=window).astype(np.float32)
    slope = sources["slope"].read(1, window=window).astype(np.float32)
    distance = sources["distance"].read(1, window=window).astype(np.float32)
    land_cover = sources["land_cover"].read(1, window=window)

    channels = [
        normalise_feature(elevation, normalisation["elevation"]),
        normalise_feature(slope, normalisation["slope"]),
        normalise_feature(distance, normalisation["distance_to_coast"]),
    ]
    for channel in channels:
        channel[~valid] = 0.0

    known = np.isin(land_cover, land_cover_classes)
    safe_land_cover = np.where(known, land_cover, 0)
    for class_value in land_cover_classes:
        channels.append(
            ((safe_land_cover == class_value) & valid).astype(np.float32)
        )

    return np.stack(channels, axis=0).astype(np.float32), valid


def batches(items: list[tuple[int, int]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_outputs(
    probability_sum: np.memmap,
    probability_weight: np.memmap,
    reference_profile: dict,
    operational_threshold: float,
    known_land_cover_classes: list[int],
) -> dict:
    height, width = probability_sum.shape
    totals = {
        "valid_land_cells": 0,
        "evaluated_cells": 0,
        "uncovered_valid_cells": 0,
        "susceptible_cells": 0,
        "uncertain_0_40_to_0_60_cells": 0,
        "unseen_land_cover_cells": 0,
        "probability_sum": 0.0,
        "uncertainty_sum": 0.0,
    }
    unseen_classes: set[int] = set()

    with ExitStack() as stack:
        valid_source = stack.enter_context(rasterio.open(VALID_MASK_PATH))
        land_cover_source = stack.enter_context(rasterio.open(LAND_COVER_PATH))
        probability_destination = stack.enter_context(
            rasterio.open(
                PROBABILITY_PATH,
                "w",
                **output_profile(reference_profile, "float32", FLOAT_NODATA),
            )
        )
        binary_destination = stack.enter_context(
            rasterio.open(
                BINARY_PATH,
                "w",
                **output_profile(reference_profile, "uint8", BYTE_NODATA),
            )
        )
        uncertainty_destination = stack.enter_context(
            rasterio.open(
                UNCERTAINTY_PATH,
                "w",
                **output_profile(reference_profile, "float32", FLOAT_NODATA),
            )
        )
        coverage_destination = stack.enter_context(
            rasterio.open(
                COVERAGE_PATH,
                "w",
                **output_profile(reference_profile, "uint8", BYTE_NODATA),
            )
        )

        windows = list(valid_source.block_windows(1))
        for sequence, (_, window) in enumerate(windows, start=1):
            row_start = int(window.row_off)
            row_end = row_start + int(window.height)
            column_start = int(window.col_off)
            column_end = column_start + int(window.width)
            region = np.s_[row_start:row_end, column_start:column_end]

            valid = valid_source.read(1, window=window) == 1
            weights = np.asarray(probability_weight[region])
            evaluated = valid & (weights > 0)

            probability = np.full(valid.shape, FLOAT_NODATA, dtype=np.float32)
            probability[evaluated] = (
                np.asarray(probability_sum[region])[evaluated] / weights[evaluated]
            )

            binary = np.full(valid.shape, BYTE_NODATA, dtype=np.uint8)
            binary[evaluated] = (
                probability[evaluated] >= operational_threshold
            ).astype(np.uint8)

            uncertainty = np.full(valid.shape, FLOAT_NODATA, dtype=np.float32)
            if np.any(evaluated):
                clipped = np.clip(probability[evaluated], 1e-6, 1.0 - 1e-6)
                uncertainty[evaluated] = (
                    -(
                        clipped * np.log(clipped)
                        + (1.0 - clipped) * np.log(1.0 - clipped)
                    )
                    / np.log(2.0)
                ).astype(np.float32)

            coverage = np.full(valid.shape, BYTE_NODATA, dtype=np.uint8)
            coverage[~valid] = 0
            coverage[valid & ~evaluated] = 2
            coverage[evaluated] = 1

            probability_destination.write(probability, 1, window=window)
            binary_destination.write(binary, 1, window=window)
            uncertainty_destination.write(uncertainty, 1, window=window)
            coverage_destination.write(coverage, 1, window=window)

            land_cover = land_cover_source.read(1, window=window)
            unseen = valid & ~np.isin(land_cover, known_land_cover_classes)
            if np.any(unseen):
                unseen_classes.update(int(value) for value in np.unique(land_cover[unseen]))

            totals["valid_land_cells"] += int(np.sum(valid))
            totals["evaluated_cells"] += int(np.sum(evaluated))
            totals["uncovered_valid_cells"] += int(np.sum(valid & ~evaluated))
            totals["susceptible_cells"] += int(np.sum(binary[evaluated] == 1))
            totals["uncertain_0_40_to_0_60_cells"] += int(
                np.sum(
                    (probability[evaluated] >= 0.40)
                    & (probability[evaluated] <= 0.60)
                )
            )
            totals["unseen_land_cover_cells"] += int(np.sum(unseen))
            totals["probability_sum"] += float(
                np.sum(probability[evaluated], dtype=np.float64)
            )
            totals["uncertainty_sum"] += float(
                np.sum(uncertainty[evaluated], dtype=np.float64)
            )

            if sequence % 100 == 0 or sequence == len(windows):
                print(f"  Output blocks written: {sequence}/{len(windows)}")

    totals["unseen_land_cover_classes"] = sorted(unseen_classes)
    return totals


def read_preview(path: Path, maximum_size: int, resampling) -> np.ma.MaskedArray:
    with rasterio.open(path) as source:
        scale = max(source.width, source.height) / maximum_size
        output_height = max(1, int(round(source.height / max(scale, 1.0))))
        output_width = max(1, int(round(source.width / max(scale, 1.0))))
        return source.read(
            1,
            out_shape=(output_height, output_width),
            masked=True,
            resampling=resampling,
        )


def create_evidence_figure(operational_threshold: float) -> None:
    elevation = read_preview(ELEVATION_PATH, 1600, Resampling.bilinear)
    probability = read_preview(PROBABILITY_PATH, 1600, Resampling.bilinear)
    binary = read_preview(BINARY_PATH, 1600, Resampling.nearest)
    uncertainty = read_preview(UNCERTAINTY_PATH, 1600, Resampling.bilinear)

    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    figure.suptitle(
        "ClimateAdapt AI – Full Nelson V2 Susceptibility Screening",
        fontsize=18,
        fontweight="bold",
    )

    image = axes[0, 0].imshow(elevation, cmap="terrain")
    axes[0, 0].set_title("Elevation and inference coverage")
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Metres")

    image = axes[0, 1].imshow(probability, cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("Predicted susceptibility probability")
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Probability")

    axes[1, 0].imshow(
        binary,
        cmap=ListedColormap(["#e8e8e8", "#d62728"]),
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(
        f"Operational susceptibility (frozen threshold = {operational_threshold:.2f})"
    )

    image = axes[1, 1].imshow(uncertainty, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("Normalised entropy uncertainty")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Uncertainty")

    for axis in axes.flat:
        axis.set_axis_off()

    figure.text(
        0.5,
        0.005,
        (
            "Frozen Experiment 2 model; label-blind regional screening. "
            "White/blank areas have no valid model input coverage."
        ),
        ha="center",
        fontsize=11,
    )
    figure.savefig(EVIDENCE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    start_time = time.perf_counter()
    required_paths = [
        ELEVATION_PATH,
        SLOPE_PATH,
        DISTANCE_PATH,
        LAND_COVER_PATH,
        VALID_MASK_PATH,
        TILE_METADATA_PATH,
        CHECKPOINT_PATH,
        THRESHOLD_PATH,
        SRC_DIR / "climateadapt" / "models" / "unet_v2.py",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))

    free_disk_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    if free_disk_gb < MINIMUM_FREE_DISK_GB:
        raise RuntimeError(
            f"At least {MINIMUM_FREE_DISK_GB:.1f} GB free disk space is required; "
            f"only {free_disk_gb:.2f} GB is available."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    raster_paths = [
        ELEVATION_PATH,
        SLOPE_PATH,
        DISTANCE_PATH,
        LAND_COVER_PATH,
        VALID_MASK_PATH,
    ]
    alignment = validate_raster_alignment(raster_paths)
    height = int(alignment["height"])
    width = int(alignment["width"])
    shape = (height, width)

    tile_metadata = load_json(TILE_METADATA_PATH)
    threshold_record = load_json(THRESHOLD_PATH)
    operational_threshold = float(
        threshold_record.get(
            "operational_threshold", threshold_record.get("selected_threshold")
        )
    )
    if not 0.0 <= operational_threshold <= 1.0:
        raise ValueError(f"Invalid operational threshold: {operational_threshold}")

    land_cover_classes = [int(value) for value in tile_metadata["land_cover_classes"]]
    expected_channels = 3 + len(land_cover_classes)
    if expected_channels != int(tile_metadata["number_of_channels"]):
        raise ValueError("Tile metadata contains an inconsistent channel definition.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict(extract_state_dict(checkpoint))
    inferred_channels, inferred_base_channels = infer_dimensions(state_dict)
    input_channels = int(checkpoint.get("input_channels", inferred_channels))
    base_channels = int(checkpoint.get("base_channels", inferred_base_channels))
    dropout_probability = float(checkpoint.get("dropout_probability", 0.20))
    if input_channels != expected_channels:
        raise ValueError(
            f"Checkpoint expects {input_channels} channels, but training metadata "
            f"defines {expected_channels}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = device.type == "cuda"
    batch_size = DEFAULT_BATCH_SIZE if device.type == "cuda" else 1
    model = CompactUNetV2(
        input_channels=input_channels,
        base_channels=base_channels,
        dropout_probability=dropout_probability,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tile_size = int(tile_metadata["tile_size_pixels"])
    stride = int(tile_metadata["stride_pixels"])
    row_positions = generate_positions(height, tile_size, stride)
    column_positions = generate_positions(width, tile_size, stride)

    candidate_positions: list[tuple[int, int]] = []
    with rasterio.open(VALID_MASK_PATH) as valid_source:
        for row in row_positions:
            for column in column_positions:
                window = Window(column, row, tile_size, tile_size)
                if np.any(valid_source.read(1, window=window) == 1):
                    candidate_positions.append((row, column))

    if not candidate_positions:
        raise RuntimeError("No inference tiles intersect valid land coverage.")

    probability_sum_path = TEMP_DIR / "probability_sum.float32.dat"
    probability_weight_path = TEMP_DIR / "probability_weight.float32.dat"
    probability_sum = create_memmap(probability_sum_path, shape)
    probability_weight = create_memmap(probability_weight_path, shape)
    weights = blend_weights(tile_size)

    print("\n" + "=" * 80)
    print("CLIMATEADAPT EXPERIMENT 2 – FULL NELSON REGION INFERENCE")
    print("=" * 80)
    print("Mode: label-blind regional susceptibility screening")
    print(f"Grid: {width:,} columns x {height:,} rows")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Frozen operational threshold: {operational_threshold:.2f}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
    print(f"Mixed precision: {mixed_precision}")
    print(f"Inference batch size: {batch_size}")
    print(f"Input channels: {input_channels}")
    print(f"Tile size / stride: {tile_size} / {stride} pixels")
    print(f"Candidate tiles: {len(candidate_positions):,}")
    print("Flood labels are not being opened.")

    source_paths = {
        "elevation": ELEVATION_PATH,
        "slope": SLOPE_PATH,
        "distance": DISTANCE_PATH,
        "land_cover": LAND_COVER_PATH,
        "valid": VALID_MASK_PATH,
    }
    processed = 0
    with ExitStack() as stack:
        sources = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in source_paths.items()
        }
        with torch.inference_mode():
            for position_batch in batches(candidate_positions, batch_size):
                feature_batch = []
                masks = []
                for row, column in position_batch:
                    window = Window(column, row, tile_size, tile_size)
                    features, valid = read_features(
                        window,
                        sources,
                        tile_metadata["normalization"],
                        land_cover_classes,
                    )
                    feature_batch.append(features)
                    masks.append(valid)

                tensor = torch.from_numpy(np.stack(feature_batch)).to(
                    device, non_blocking=True
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    enabled=mixed_precision,
                ):
                    predictions = torch.sigmoid(model(tensor))
                predictions = predictions[:, 0].float().cpu().numpy()

                for (row, column), probability, valid in zip(
                    position_batch, predictions, masks
                ):
                    row_end = row + tile_size
                    column_end = column + tile_size
                    region = np.s_[row:row_end, column:column_end]
                    valid_weights = weights * valid.astype(np.float32)
                    probability_sum[region] += probability * valid_weights
                    probability_weight[region] += valid_weights

                processed += len(position_batch)
                if processed % 250 < batch_size or processed == len(candidate_positions):
                    print(f"  Inference tiles processed: {processed:,}/{len(candidate_positions):,}")

    probability_sum.flush()
    probability_weight.flush()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print("\nWriting full-region output rasters...")
    with rasterio.open(ELEVATION_PATH) as reference_source:
        reference_profile = reference_source.profile.copy()
    totals = write_outputs(
        probability_sum,
        probability_weight,
        reference_profile,
        operational_threshold,
        land_cover_classes,
    )

    del probability_sum, probability_weight
    gc.collect()
    probability_sum_path.unlink(missing_ok=True)
    probability_weight_path.unlink(missing_ok=True)
    try:
        TEMP_DIR.rmdir()
    except OSError:
        pass

    evaluated_cells = int(totals["evaluated_cells"])
    if evaluated_cells == 0:
        raise RuntimeError("No evaluated full-region cells were produced.")

    cell_area_m2 = abs(
        float(reference_profile["transform"].a)
        * float(reference_profile["transform"].e)
    )
    evaluated_area_km2 = evaluated_cells * cell_area_m2 / 1_000_000.0
    susceptible_cells = int(totals["susceptible_cells"])
    susceptible_area_km2 = susceptible_cells * cell_area_m2 / 1_000_000.0
    susceptible_percentage = 100.0 * susceptible_cells / evaluated_cells
    mean_probability = float(totals["probability_sum"] / evaluated_cells)
    mean_uncertainty = float(totals["uncertainty_sum"] / evaluated_cells)
    processing_minutes = (time.perf_counter() - start_time) / 60.0

    summary = {
        "experiment": "Experiment 2 – GroupNorm Compact U-Net",
        "analysis_type": "label-blind full-region planning screening",
        "model_retrained": False,
        "threshold_modified": False,
        "flood_target_raster_accessed": False,
        "threshold_source": threshold_record.get("selection_dataset"),
        "operational_threshold": operational_threshold,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "mixed_precision": mixed_precision,
        "batch_size": batch_size,
        "input_channels": input_channels,
        "grid_width": width,
        "grid_height": height,
        "tile_size_pixels": tile_size,
        "stride_pixels": stride,
        "candidate_tiles": len(candidate_positions),
        "overlap_blending": "raised-cosine weighted probability mean",
        "valid_land_cells": int(totals["valid_land_cells"]),
        "evaluated_cells": evaluated_cells,
        "uncovered_valid_cells": int(totals["uncovered_valid_cells"]),
        "evaluated_area_square_kilometres": evaluated_area_km2,
        "susceptible_cells": susceptible_cells,
        "susceptible_area_square_kilometres": susceptible_area_km2,
        "susceptible_percentage": susceptible_percentage,
        "mean_probability": mean_probability,
        "mean_normalised_entropy": mean_uncertainty,
        "probability_0_40_to_0_60_cells": int(
            totals["uncertain_0_40_to_0_60_cells"]
        ),
        "training_land_cover_classes": land_cover_classes,
        "unseen_land_cover_classes": totals["unseen_land_cover_classes"],
        "unseen_land_cover_cells": int(totals["unseen_land_cover_cells"]),
        "unseen_land_cover_handling": (
            "Mapped to training class-0 channel; model weights and channel count unchanged."
        ),
        "processing_time_minutes": processing_minutes,
        "interpretation_warning": (
            "Screening-level susceptibility only; not a validated flood extent, "
            "engineering design product, or emergency-warning service."
        ),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(
            {"metric": key, "value": value} for key, value in summary.items()
        )

    print("Creating regional evidence figure...")
    create_evidence_figure(operational_threshold)

    print("\n" + "=" * 80)
    print("FULL NELSON REGION INFERENCE COMPLETED")
    print("=" * 80)
    print(f"Evaluated land cells:        {evaluated_cells:,}")
    print(f"Uncovered valid cells:       {int(totals['uncovered_valid_cells']):,}")
    print(f"Evaluated land area:         {evaluated_area_km2:.3f} km²")
    print(f"Predicted susceptible cells: {susceptible_cells:,}")
    print(f"Predicted susceptible area:  {susceptible_area_km2:.3f} km²")
    print(f"Susceptible percentage:      {susceptible_percentage:.2f}%")
    print(f"Mean probability:            {mean_probability:.4f}")
    print(f"Mean uncertainty:            {mean_uncertainty:.4f}")
    print(f"Unseen LCDB classes:         {totals['unseen_land_cover_classes']}")
    print(f"Processing time:             {processing_minutes:.2f} minutes")
    print("-" * 80)
    print(f"Probability: {PROBABILITY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Binary:      {BINARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Uncertainty: {UNCERTAINTY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Coverage:    {COVERAGE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary:     {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Figure:      {EVIDENCE_FIGURE.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    print("The frozen model and threshold were not changed.")
    print("Flood-label rasters were not accessed.")


if __name__ == "__main__":
    main()
