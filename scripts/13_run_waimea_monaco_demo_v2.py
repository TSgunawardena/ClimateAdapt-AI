"""Run the frozen Experiment 2 model on Waimea inlet / Monaco.

This is a label-blind demonstration. The script deliberately does not open the
flood-target raster, calculate accuracy metrics, retrain the model, or adjust
the validation-selected operational threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.models.unet_v2 import CompactUNetV2  # noqa: E402


RASTER_DIR = PROJECT_ROOT / "data" / "processed" / "combined_rasters"
TILE_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "tiles" / "tile_metadata.json"
)
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "climateadapt_unet_v2_best.pt"
THRESHOLD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation_v2"
    / "selected_threshold_v2.json"
)

ELEVATION_PATH = RASTER_DIR / "elevation_5m.tif"
SLOPE_PATH = RASTER_DIR / "slope_degrees_5m.tif"
DISTANCE_PATH = RASTER_DIR / "distance_to_coast_5m.tif"
LAND_COVER_PATH = RASTER_DIR / "land_cover_class_5m.tif"
VALID_MASK_PATH = RASTER_DIR / "valid_land_mask_5m.tif"
SPLIT_MASK_PATH = RASTER_DIR / "split_mask_5m.tif"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demonstration_v2"
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures"

PROBABILITY_RASTER = OUTPUT_DIR / "waimea_monaco_probability_v2.tif"
BINARY_RASTER = OUTPUT_DIR / "waimea_monaco_susceptibility_v2.tif"
UNCERTAINTY_RASTER = OUTPUT_DIR / "waimea_monaco_uncertainty_v2.tif"
SUMMARY_JSON = OUTPUT_DIR / "waimea_monaco_demo_summary_v2.json"
SUMMARY_CSV = OUTPUT_DIR / "waimea_monaco_demo_summary_v2.csv"
EVIDENCE_FIGURE = FIGURE_DIR / "waimea_monaco_demonstration_v2.png"

DEMONSTRATION_SPLIT_ID = 4


def read_float_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        array = source.read(1).astype(np.float32)
        nodata = source.nodata
        profile = source.profile.copy()
    if nodata is not None:
        array[array == nodata] = np.nan
    return array, profile


def read_integer_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read(1)


def normalise_feature(array: np.ndarray, statistics: dict) -> np.ndarray:
    working = array.astype(np.float32, copy=True)
    if bool(statistics.get("log1p_transformation", False)):
        working = np.log1p(np.maximum(working, 0.0))

    lower = float(statistics["lower_percentile"])
    upper = float(statistics["upper_percentile"])
    mean = float(statistics["mean"])
    standard_deviation = max(float(statistics["standard_deviation"]), 1e-6)

    working = np.clip(working, lower, upper)
    working = (working - mean) / standard_deviation
    working[~np.isfinite(working)] = 0.0
    return working.astype(np.float32)


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
    weights = [
        value
        for value in state_dict.values()
        if torch.is_tensor(value) and value.ndim == 4
    ]
    if not weights:
        raise RuntimeError("No convolution weights were found in the checkpoint.")
    return int(weights[0].shape[1]), int(weights[0].shape[0])


def save_raster(
    path: Path,
    array: np.ndarray,
    reference_profile: dict,
    dtype: str,
    nodata,
) -> None:
    profile = reference_profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array.astype(dtype), 1)


def masked_crop(mask: np.ndarray, padding: int = 20) -> tuple[slice, slice]:
    rows, columns = np.where(mask)
    if len(rows) == 0:
        raise ValueError("The demonstration split contains no valid land cells.")
    return (
        slice(max(0, int(rows.min()) - padding), min(mask.shape[0], int(rows.max()) + padding + 1)),
        slice(
            max(0, int(columns.min()) - padding),
            min(mask.shape[1], int(columns.max()) + padding + 1),
        ),
    )


def plot_extent(transform, row_slice: slice, column_slice: slice) -> list[float]:
    left, top = transform * (column_slice.start, row_slice.start)
    right, bottom = transform * (column_slice.stop, row_slice.stop)
    return [left, right, bottom, top]


def main() -> None:
    required = [
        ELEVATION_PATH,
        SLOPE_PATH,
        DISTANCE_PATH,
        LAND_COVER_PATH,
        VALID_MASK_PATH,
        SPLIT_MASK_PATH,
        TILE_METADATA_PATH,
        CHECKPOINT_PATH,
        THRESHOLD_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with TILE_METADATA_PATH.open("r", encoding="utf-8") as file:
        tile_metadata = json.load(file)
    with THRESHOLD_PATH.open("r", encoding="utf-8") as file:
        threshold_record = json.load(file)

    operational_threshold = float(
        threshold_record.get(
            "operational_threshold",
            threshold_record.get("selected_threshold"),
        )
    )
    if not 0.0 <= operational_threshold <= 1.0:
        raise ValueError(f"Invalid operational threshold: {operational_threshold}")
    if bool(threshold_record.get("demonstration_dataset_used", False)):
        raise RuntimeError(
            "The threshold record says demonstration data were previously used."
        )

    elevation, reference_profile = read_float_raster(ELEVATION_PATH)
    slope, _ = read_float_raster(SLOPE_PATH)
    distance, _ = read_float_raster(DISTANCE_PATH)
    land_cover = read_integer_raster(LAND_COVER_PATH)
    valid_land = read_integer_raster(VALID_MASK_PATH)
    split_mask = read_integer_raster(SPLIT_MASK_PATH)

    expected_shape = elevation.shape
    arrays = {
        "slope": slope,
        "distance": distance,
        "land cover": land_cover,
        "valid-land mask": valid_land,
        "split mask": split_mask,
    }
    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape {array.shape} does not match {expected_shape}."
            )

    demonstration_mask = (valid_land == 1) & (
        split_mask == DEMONSTRATION_SPLIT_ID
    )
    demonstration_cells = int(np.sum(demonstration_mask))
    if demonstration_cells == 0:
        raise ValueError("No valid split-4 demonstration cells were found.")

    normalisation = tile_metadata["normalization"]
    continuous = np.stack(
        [
            normalise_feature(elevation, normalisation["elevation"]),
            normalise_feature(slope, normalisation["slope"]),
            normalise_feature(distance, normalisation["distance_to_coast"]),
        ],
        axis=0,
    )
    land_cover_classes = [int(value) for value in tile_metadata["land_cover_classes"]]
    expected_channels = 3 + len(land_cover_classes)
    metadata_channels = int(tile_metadata["number_of_channels"])
    if expected_channels != metadata_channels:
        raise ValueError(
            f"Metadata channel mismatch: calculated {expected_channels}, "
            f"recorded {metadata_channels}."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict(extract_state_dict(checkpoint))
    inferred_input_channels, inferred_base_channels = infer_dimensions(state_dict)
    input_channels = int(checkpoint.get("input_channels", inferred_input_channels))
    base_channels = int(checkpoint.get("base_channels", inferred_base_channels))
    dropout_probability = float(checkpoint.get("dropout_probability", 0.20))
    if input_channels != expected_channels:
        raise ValueError(
            f"Checkpoint expects {input_channels} channels but preprocessing "
            f"metadata defines {expected_channels}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactUNetV2(
        input_channels=input_channels,
        base_channels=base_channels,
        dropout_probability=dropout_probability,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    tile_size = int(tile_metadata["tile_size_pixels"])
    stride = int(tile_metadata["stride_pixels"])
    height, width = expected_shape
    row_positions = generate_positions(height, tile_size, stride)
    column_positions = generate_positions(width, tile_size, stride)

    candidate_positions = []
    for row in row_positions:
        for column in column_positions:
            tile_mask = demonstration_mask[
                row : row + tile_size,
                column : column + tile_size,
            ]
            if np.any(tile_mask):
                candidate_positions.append((row, column))

    if not candidate_positions:
        raise RuntimeError("No demonstration inference tiles were generated.")

    probability_sum = np.zeros(expected_shape, dtype=np.float32)
    probability_weight = np.zeros(expected_shape, dtype=np.float32)

    # Raised-cosine blending gives the reliable centre of each tile more
    # influence than its padded edges. A small non-zero floor preserves full
    # coverage along the outside boundary of the demonstration region.
    blend_axis = np.linspace(-1.0, 1.0, tile_size, dtype=np.float32)
    blend_axis = 0.05 + 0.95 * 0.5 * (
        1.0 + np.cos(np.pi * blend_axis)
    )
    blend_weights = np.outer(blend_axis, blend_axis).astype(np.float32)

    print("\n" + "=" * 78)
    print("CLIMATEADAPT EXPERIMENT 2 – WAIMEA / MONACO DEMONSTRATION")
    print("=" * 78)
    print("Mode: label-blind susceptibility demonstration")
    print(f"Frozen operational threshold: {operational_threshold:.2f}")
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Device: {device}")
    print(f"Input channels: {input_channels}")
    print(f"Demonstration cells: {demonstration_cells:,}")
    print(f"Inference tiles: {len(candidate_positions)}")
    print("Running inference without loading demonstration labels...")

    with torch.inference_mode():
        for sequence, (row, column) in enumerate(candidate_positions, start=1):
            row_end = row + tile_size
            column_end = column + tile_size
            region_mask = demonstration_mask[row:row_end, column:column_end]

            channels = []
            for index in range(continuous.shape[0]):
                channel = continuous[index, row:row_end, column:column_end].copy()
                channel[~region_mask] = 0.0
                channels.append(channel)

            land_cover_tile = land_cover[row:row_end, column:column_end]
            for class_value in land_cover_classes:
                channels.append(
                    ((land_cover_tile == class_value) & region_mask).astype(np.float32)
                )

            features = np.stack(channels, axis=0).astype(np.float32)
            tensor = torch.from_numpy(features[None]).to(device)
            probability = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

            destination = np.s_[row:row_end, column:column_end]
            valid_weights = blend_weights * region_mask.astype(np.float32)
            probability_sum[destination] += probability * valid_weights
            probability_weight[destination] += valid_weights

            if sequence % 25 == 0 or sequence == len(candidate_positions):
                print(
                    f"Processed demonstration tile {sequence}/"
                    f"{len(candidate_positions)}"
                )

    evaluated = demonstration_mask & (probability_weight > 0)
    evaluated_cells = int(np.sum(evaluated))
    uncovered_cells = int(np.sum(demonstration_mask & ~evaluated))
    if evaluated_cells == 0:
        raise RuntimeError("No valid demonstration predictions were produced.")

    probability = np.full(expected_shape, np.nan, dtype=np.float32)
    probability[evaluated] = (
        probability_sum[evaluated] / probability_weight[evaluated]
    )
    binary = np.zeros(expected_shape, dtype=np.uint8)
    binary[evaluated] = (probability[evaluated] >= operational_threshold).astype(
        np.uint8
    )

    # Normalised binary entropy: 0=certain, 1=maximum uncertainty at p=0.5.
    clipped_probability = np.clip(probability[evaluated], 1e-6, 1.0 - 1e-6)
    entropy = -(
        clipped_probability * np.log(clipped_probability)
        + (1.0 - clipped_probability) * np.log(1.0 - clipped_probability)
    ) / np.log(2.0)
    uncertainty = np.full(expected_shape, np.nan, dtype=np.float32)
    uncertainty[evaluated] = entropy.astype(np.float32)

    susceptible_cells = int(np.sum(binary[evaluated] == 1))
    cell_area_m2 = abs(
        float(reference_profile["transform"].a)
        * float(reference_profile["transform"].e)
    )
    susceptible_area_km2 = susceptible_cells * cell_area_m2 / 1_000_000.0
    evaluated_area_km2 = evaluated_cells * cell_area_m2 / 1_000_000.0
    susceptible_percentage = 100.0 * susceptible_cells / evaluated_cells
    mean_probability = float(np.mean(probability[evaluated]))
    median_probability = float(np.median(probability[evaluated]))
    mean_uncertainty = float(np.mean(uncertainty[evaluated]))
    high_uncertainty_cells = int(
        np.sum(
            (probability[evaluated] >= 0.40)
            & (probability[evaluated] <= 0.60)
        )
    )

    probability_output = np.full(expected_shape, -9999.0, dtype=np.float32)
    uncertainty_output = np.full(expected_shape, -9999.0, dtype=np.float32)
    binary_output = np.full(expected_shape, 255, dtype=np.uint8)
    probability_output[evaluated] = probability[evaluated]
    uncertainty_output[evaluated] = uncertainty[evaluated]
    binary_output[evaluated] = binary[evaluated]

    save_raster(
        PROBABILITY_RASTER,
        probability_output,
        reference_profile,
        "float32",
        -9999.0,
    )
    save_raster(
        BINARY_RASTER,
        binary_output,
        reference_profile,
        "uint8",
        255,
    )
    save_raster(
        UNCERTAINTY_RASTER,
        uncertainty_output,
        reference_profile,
        "float32",
        -9999.0,
    )

    summary = {
        "experiment": "Experiment 2 – GroupNorm Compact U-Net",
        "analysis_type": "label-blind planning demonstration",
        "demonstration_area": "Waimea inlet and Monaco",
        "demonstration_split_id": DEMONSTRATION_SPLIT_ID,
        "flood_target_raster_accessed": False,
        "model_retrained": False,
        "threshold_modified": False,
        "threshold_source": "Rocks Road validation region",
        "operational_threshold": operational_threshold,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_channels": input_channels,
        "inference_tiles": len(candidate_positions),
        "overlap_blending": "raised-cosine weighted probability mean",
        "demonstration_cells": demonstration_cells,
        "evaluated_cells": evaluated_cells,
        "uncovered_cells": uncovered_cells,
        "evaluated_area_square_kilometres": evaluated_area_km2,
        "susceptible_cells": susceptible_cells,
        "susceptible_area_square_kilometres": susceptible_area_km2,
        "susceptible_percentage": susceptible_percentage,
        "mean_probability": mean_probability,
        "median_probability": median_probability,
        "mean_normalised_entropy": mean_uncertainty,
        "probability_0_40_to_0_60_cells": high_uncertainty_cells,
        "interpretation_warning": (
            "Susceptibility output for decision-support demonstration only; "
            "not a validated flood extent or emergency-warning product."
        ),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    rows = [{"metric": key, "value": value} for key, value in summary.items()]
    import csv

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    row_slice, column_slice = masked_crop(evaluated)
    transform = reference_profile["transform"]
    extent = plot_extent(transform, row_slice, column_slice)
    local_mask = evaluated[row_slice, column_slice]

    elevation_plot = np.ma.masked_where(
        ~local_mask, elevation[row_slice, column_slice]
    )
    probability_plot = np.ma.masked_where(
        ~local_mask, probability[row_slice, column_slice]
    )
    binary_plot = np.ma.masked_where(~local_mask, binary[row_slice, column_slice])
    uncertainty_plot = np.ma.masked_where(
        ~local_mask, uncertainty[row_slice, column_slice]
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    figure.suptitle(
        "ClimateAdapt AI – Waimea Inlet / Monaco Demonstration",
        fontsize=18,
        fontweight="bold",
    )

    elevation_image = axes[0, 0].imshow(
        elevation_plot,
        extent=extent,
        origin="upper",
        cmap="terrain",
    )
    axes[0, 0].set_title("Elevation input")
    figure.colorbar(
        elevation_image, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Metres"
    )

    probability_image = axes[0, 1].imshow(
        probability_plot,
        extent=extent,
        origin="upper",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title("Predicted susceptibility probability")
    figure.colorbar(
        probability_image,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
        label="Probability",
    )

    axes[1, 0].imshow(
        binary_plot,
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#f2f2f2", "#d62728"]),
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(
        f"Operational susceptibility (frozen threshold = {operational_threshold:.2f})"
    )

    uncertainty_image = axes[1, 1].imshow(
        uncertainty_plot,
        extent=extent,
        origin="upper",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title("Predictive uncertainty indicator")
    figure.colorbar(
        uncertainty_image,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
        label="Normalised entropy",
    )

    for axis in axes.flat:
        axis.set_xlabel("Easting (NZTM2000)")
        axis.set_ylabel("Northing (NZTM2000)")
        axis.ticklabel_format(style="plain", useOffset=False)

    figure.text(
        0.5,
        0.005,
        (
            f"Predicted susceptible area: {susceptible_area_km2:.3f} km² "
            f"({susceptible_percentage:.1f}% of evaluated land) | "
            "Label-blind demonstration; no accuracy claim"
        ),
        ha="center",
        fontsize=11,
    )
    figure.savefig(EVIDENCE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("\n" + "=" * 78)
    print("WAIMEA / MONACO DEMONSTRATION COMPLETED")
    print("=" * 78)
    print(f"Evaluated cells:             {evaluated_cells:,}")
    print(f"Uncovered cells:             {uncovered_cells:,}")
    print(f"Evaluated land area:         {evaluated_area_km2:.3f} km²")
    print(f"Predicted susceptible cells: {susceptible_cells:,}")
    print(f"Predicted susceptible area:  {susceptible_area_km2:.3f} km²")
    print(f"Susceptible percentage:      {susceptible_percentage:.2f}%")
    print(f"Mean probability:            {mean_probability:.4f}")
    print(f"Median probability:          {median_probability:.4f}")
    print(f"Mean uncertainty:            {mean_uncertainty:.4f}")
    print(f"Probability 0.40–0.60 cells: {high_uncertainty_cells:,}")
    print("-" * 78)
    print(f"Summary JSON: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Summary CSV: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Probability raster: {PROBABILITY_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Binary raster: {BINARY_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Uncertainty raster: {UNCERTAINTY_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Evidence figure: {EVIDENCE_FIGURE.relative_to(PROJECT_ROOT)}")
    print("=" * 78)
    print("The flood-target raster was not accessed.")
    print("No model weights or thresholds were changed.")
    print("This output is a planning demonstration, not a validated flood extent.")


if __name__ == "__main__":
    main()
