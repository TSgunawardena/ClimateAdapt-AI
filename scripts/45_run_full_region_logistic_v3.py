"""Run frozen ClimateAdapt V3 logistic inference over the full Nelson region.

The script is deliberately label-blind: it never opens a flood-target or
flood-area raster. It reconstructs the exact 11 V3 feature channels, validates
them against a saved training tile, and processes the regional rasters in
windows so it can run on an ordinary workstation.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RASTER_DIR = PROJECT_ROOT / "data" / "processed" / "full_region" / "rasters"
V3_TILES_DIR = PROJECT_ROOT / "data" / "processed" / "v3" / "tiles"

ELEVATION_PATH = RASTER_DIR / "elevation_5m.tif"
SLOPE_PATH = RASTER_DIR / "slope_degrees_5m.tif"
DISTANCE_PATH = RASTER_DIR / "distance_to_coast_5m.tif"
LAND_COVER_PATH = RASTER_DIR / "land_cover_class_5m.tif"
VALID_MASK_PATH = RASTER_DIR / "valid_land_mask_5m.tif"
TILE_METADATA_PATH = V3_TILES_DIR / "v3_tile_metadata.json"
TILE_MANIFEST_PATH = V3_TILES_DIR / "v3_tile_manifest.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "v3" / "logistic_regression_v3.joblib"
BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "baselines_v3" / "baseline_validation_summary_v3.json"
)
OPTIONAL_THRESHOLD_PATH = (
    PROJECT_ROOT / "outputs" / "baselines_v3" / "logistic_selected_threshold_v3.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_region_inference_v3"
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures" / "v3"
SCORE_RASTER = OUTPUT_DIR / "nelson_logistic_susceptibility_score_v3.tif"
BINARY_RASTER = OUTPUT_DIR / "nelson_logistic_susceptibility_v3.tif"
COVERAGE_RASTER = OUTPUT_DIR / "nelson_logistic_coverage_v3.tif"
SUMMARY_JSON = OUTPUT_DIR / "nelson_logistic_full_region_summary_v3.json"
SUMMARY_CSV = OUTPUT_DIR / "nelson_logistic_full_region_summary_v3.csv"
FIGURE_PATH = FIGURE_DIR / "nelson_full_region_logistic_v3.png"

INPUT_CHANNELS = 11
BLOCK_SIZE = 512
FALLBACK_THRESHOLD = 0.004
DEFAULT_NORMALISATION = {
    "elevation": {
        "lower": 0.376,
        "upper": 192.664,
        "mean": 28.187,
        "std": 41.139,
    },
    "slope": {
        "lower": 0.0,
        "upper": 40.391,
        "mean": 8.004,
        "std": 9.946,
    },
    "log1p_distance_to_coast": {
        "lower": 1.792,
        "upper": 7.557,
        "mean": 5.861,
        "std": 1.165,
    },
}

# Channels 3..10 use these eight semantic groups in this exact order.
LAND_COVER_GROUPS = {
    0: {0},
    1: {1, 2, 5, 6},
    2: {10, 12, 16, 64},
    3: {20, 21, 22},
    4: {30, 33, 40, 41, 43},
    5: {45, 46},
    6: {50, 51, 52, 55, 56, 58},
    7: {54, 68, 69, 71},
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_nested(mapping, names: tuple[str, ...], default=None):
    if isinstance(mapping, dict):
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
        for value in mapping.values():
            result = find_nested(value, names, None)
            if result is not None:
                return result
    return default


def unwrap_estimator(artifact):
    if hasattr(artifact, "predict_proba"):
        return artifact, {}
    if isinstance(artifact, dict):
        for key in ("model", "pipeline", "estimator", "classifier"):
            candidate = artifact.get(key)
            if hasattr(candidate, "predict_proba"):
                return candidate, artifact
    raise TypeError("The V3 logistic artifact has no predict_proba estimator.")


def canonical_feature_name(name: str) -> str | None:
    text = str(name).lower().replace("-", "_").replace(" ", "_")
    if "distance" in text and "coast" in text:
        return "log1p_distance_to_coast"
    if "elev" in text or text in {"dem", "height"}:
        return "elevation"
    if "slope" in text:
        return "slope"
    return None


def normalisation_record(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    aliases = {
        "lower": ("lower", "clip_lower", "minimum", "min", "p01"),
        "upper": ("upper", "clip_upper", "maximum", "max", "p99"),
        "mean": ("mean",),
        "std": ("std", "standard_deviation", "scale"),
    }
    output = {}
    for target, names in aliases.items():
        found = None
        for name in names:
            if name in value and value[name] is not None:
                found = value[name]
                break
        if found is None:
            return None
        output[target] = float(found)
    if output["std"] <= 0 or output["upper"] < output["lower"]:
        return None
    return output


def discover_normalisation(metadata: dict) -> tuple[dict, str]:
    """Find exact training statistics regardless of minor metadata nesting."""
    discovered = {}

    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                canonical = canonical_feature_name(key)
                record = normalisation_record(value)
                if canonical and record and canonical not in discovered:
                    discovered[canonical] = record
                visit(value)
        elif isinstance(node, list):
            for value in node:
                if isinstance(value, dict):
                    name = value.get("name") or value.get("feature") or value.get("channel")
                    canonical = canonical_feature_name(name) if name else None
                    record = normalisation_record(value)
                    if canonical and record and canonical not in discovered:
                        discovered[canonical] = record
                visit(value)

    visit(metadata)
    required = set(DEFAULT_NORMALISATION)
    if required.issubset(discovered):
        return {key: discovered[key] for key in DEFAULT_NORMALISATION}, "tile metadata"
    return DEFAULT_NORMALISATION.copy(), "documented Script 40 fallback"


def normalise(values: np.ndarray, statistics: dict) -> np.ndarray:
    clipped = np.clip(values, statistics["lower"], statistics["upper"])
    return ((clipped - statistics["mean"]) / statistics["std"]).astype(np.float32)


def semantic_land_cover(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.zeros(codes.shape, dtype=np.uint8)
    recognised = np.zeros(codes.shape, dtype=bool)
    for group_id, class_codes in LAND_COVER_GROUPS.items():
        selector = np.isin(codes, list(class_codes))
        groups[selector] = group_id
        recognised |= selector
    # Truly unknown codes use group 0, matching the training unknown channel.
    groups[~recognised] = 0
    return groups, recognised


def construct_features(
    elevation: np.ndarray,
    slope: np.ndarray,
    distance: np.ndarray,
    land_cover: np.ndarray,
    normalisation: dict,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = elevation.shape
    features = np.zeros((rows, columns, INPUT_CHANNELS), dtype=np.float32)
    features[..., 0] = normalise(elevation, normalisation["elevation"])
    features[..., 1] = normalise(slope, normalisation["slope"])
    log_distance = np.log1p(np.maximum(distance, 0.0)).astype(np.float32)
    features[..., 2] = normalise(
        log_distance, normalisation["log1p_distance_to_coast"]
    )
    groups, recognised = semantic_land_cover(land_cover)
    for group_id in range(8):
        features[..., 3 + group_id] = groups == group_id
    return features, recognised


def resolve_tile_path(relative_path: str) -> Path:
    normalised = str(relative_path).replace("\\", "/")
    for candidate in (
        V3_TILES_DIR / normalised,
        PROJECT_ROOT / normalised,
        Path(normalised),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate validation tile {relative_path}")


def saved_tile_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as tile:
        for key in ("features", "image", "x"):
            if key in tile.files:
                features = np.asarray(tile[key], dtype=np.float32)
                break
        else:
            raise KeyError(f"No features found in {path.name}")
        for key in ("valid_mask", "valid"):
            if key in tile.files:
                valid = np.squeeze(np.asarray(tile[key])).astype(bool)
                break
        else:
            raise KeyError(f"No valid mask found in {path.name}")
    if features.shape[0] == INPUT_CHANNELS:
        features = np.moveaxis(features, 0, -1)
    elif features.shape[-1] != INPUT_CHANNELS:
        raise ValueError(f"Unexpected saved feature shape {features.shape}")
    return features, valid


def verify_feature_reconstruction(normalisation: dict) -> dict:
    manifest = pd.read_csv(TILE_MANIFEST_PATH)
    training = manifest[
        manifest["split"].astype(str).str.lower().str.strip().isin(("train", "training", "1"))
    ]
    if training.empty:
        raise RuntimeError("No training tile is available for feature verification.")
    row = training.iloc[0]
    path = resolve_tile_path(str(row["file"]))
    saved, saved_valid = saved_tile_features(path)
    height, width = saved_valid.shape
    window = Window(int(row["column"]), int(row["row"]), width, height)
    with (
        rasterio.open(ELEVATION_PATH) as elevation_source,
        rasterio.open(SLOPE_PATH) as slope_source,
        rasterio.open(DISTANCE_PATH) as distance_source,
        rasterio.open(LAND_COVER_PATH) as land_cover_source,
        rasterio.open(VALID_MASK_PATH) as valid_source,
    ):
        elevation = elevation_source.read(1, window=window).astype(np.float32)
        slope = slope_source.read(1, window=window).astype(np.float32)
        distance = distance_source.read(1, window=window).astype(np.float32)
        land_cover = land_cover_source.read(1, window=window)
        regional_valid = valid_source.read(1, window=window) == 1
    reconstructed, _ = construct_features(
        elevation, slope, distance, land_cover, normalisation
    )
    compare = saved_valid & regional_valid
    if not np.any(compare):
        raise RuntimeError("Feature verification tile has no comparable valid pixels.")
    difference = np.abs(saved[compare] - reconstructed[compare])
    maximum = float(np.nanmax(difference))
    mean = float(np.nanmean(difference))
    # JSON metadata should make this almost exact. The fallback accepts the
    # displayed three-decimal statistics but still rejects a wrong pipeline.
    tolerance = 0.01
    if not np.isfinite(maximum) or maximum > tolerance:
        raise RuntimeError(
            "Full-region features do not reproduce the saved V3 tile "
            f"(maximum absolute difference={maximum:.6f}, tolerance={tolerance}). "
            "Do not run regional inference until feature parity is restored."
        )
    return {
        "tile": str(path.relative_to(PROJECT_ROOT)),
        "pixels_checked": int(np.sum(compare)),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": mean,
        "tolerance": tolerance,
        "passed": True,
    }


def iter_windows(width: int, height: int, block_size: int = BLOCK_SIZE):
    for row_off in range(0, height, block_size):
        window_height = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            window_width = min(block_size, width - col_off)
            yield Window(col_off, row_off, window_width, window_height)


def output_profile(reference_profile: dict, dtype: str, nodata, predictor: int) -> dict:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="DEFLATE",
        predictor=predictor,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    return profile


def create_figure(threshold: float) -> None:
    maximum_dimension = 1800
    with rasterio.open(SCORE_RASTER) as score_source:
        scale = min(1.0, maximum_dimension / max(score_source.width, score_source.height))
        out_height = max(1, round(score_source.height * scale))
        out_width = max(1, round(score_source.width * scale))
        score = score_source.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
        )
        extent = (
            score_source.bounds.left,
            score_source.bounds.right,
            score_source.bounds.bottom,
            score_source.bounds.top,
        )
        score_nodata = score_source.nodata
    with rasterio.open(BINARY_RASTER) as binary_source:
        binary = binary_source.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        )
    score = np.ma.masked_where(score == score_nodata, score)
    binary = np.ma.masked_where(binary == 255, binary)

    figure, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    score_image = axes[0].imshow(
        score, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1
    )
    figure.colorbar(score_image, ax=axes[0], fraction=0.035, pad=0.02).set_label(
        "Model susceptibility score"
    )
    axes[0].set_title("Frozen logistic susceptibility score")
    binary_image = axes[1].imshow(
        binary,
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#d9d9d9", "#d73027"]),
        vmin=0,
        vmax=1,
    )
    colourbar = figure.colorbar(binary_image, ax=axes[1], fraction=0.035, pad=0.02)
    colourbar.set_ticks([0.25, 0.75], labels=["Below threshold", "Susceptible"])
    axes[1].set_title(f"Frozen operational threshold = {threshold:.3f}")
    for axis in axes:
        axis.set_xlabel("Easting (NZTM2000)")
        axis.set_ylabel("Northing (NZTM2000)")
        axis.set_aspect("equal")
    figure.suptitle(
        "ClimateAdapt V3 – Full Nelson Region Logistic Screening",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(FIGURE_PATH, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    required = (
        ELEVATION_PATH,
        SLOPE_PATH,
        DISTANCE_PATH,
        LAND_COVER_PATH,
        VALID_MASK_PATH,
        TILE_METADATA_PATH,
        TILE_MANIFEST_PATH,
        MODEL_PATH,
        BASELINE_SUMMARY_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V3 artifact(s):\n  " + "\n  ".join(missing))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = read_json(TILE_METADATA_PATH)
    normalisation, normalisation_source = discover_normalisation(metadata)
    baseline_summary = read_json(BASELINE_SUMMARY_PATH)
    if OPTIONAL_THRESHOLD_PATH.exists():
        threshold_information = read_json(OPTIONAL_THRESHOLD_PATH)
    else:
        threshold_information = baseline_summary
    threshold = find_nested(
        threshold_information,
        ("selected_threshold", "logistic_threshold", "operational_threshold"),
        FALLBACK_THRESHOLD,
    )
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid frozen logistic threshold: {threshold}")

    artifact = joblib.load(MODEL_PATH)
    estimator, model_metadata = unwrap_estimator(artifact)
    expected_features = int(getattr(estimator, "n_features_in_", INPUT_CHANNELS))
    if expected_features != INPUT_CHANNELS:
        raise ValueError(
            f"Logistic model expects {expected_features} features, not {INPUT_CHANNELS}."
        )

    print("\n" + "=" * 78)
    print("CLIMATEADAPT V3 – FULL NELSON LOGISTIC INFERENCE")
    print("=" * 78)
    print("Mode: label-blind regional susceptibility screening")
    print(f"Frozen threshold: {threshold:.3f}")
    print(f"Input channels: {INPUT_CHANNELS}")
    print(f"Normalisation source: {normalisation_source}")
    print("Flood-target and flood-area rasters will not be opened.")
    print("Checking exact feature parity against a saved V3 training tile...")
    feature_check = verify_feature_reconstruction(normalisation)
    print(
        f"Feature parity: PASS | max difference="
        f"{feature_check['maximum_absolute_difference']:.8f}"
    )

    with (
        rasterio.open(ELEVATION_PATH) as elevation_source,
        rasterio.open(SLOPE_PATH) as slope_source,
        rasterio.open(DISTANCE_PATH) as distance_source,
        rasterio.open(LAND_COVER_PATH) as land_cover_source,
        rasterio.open(VALID_MASK_PATH) as valid_source,
    ):
        sources = (
            elevation_source,
            slope_source,
            distance_source,
            land_cover_source,
            valid_source,
        )
        reference = elevation_source
        width, height = reference.width, reference.height
        crs = str(reference.crs)
        transform = reference.transform
        for source in sources[1:]:
            if (
                source.width != width
                or source.height != height
                or source.crs != reference.crs
                or source.transform != transform
            ):
                raise RuntimeError(f"Raster alignment mismatch: {source.name}")

        windows = list(iter_windows(width, height))
        print(f"Grid: {width:,} x {height:,}")
        print(f"Processing windows: {len(windows):,}")

        score_profile = output_profile(reference.profile, "float32", -9999.0, 2)
        binary_profile = output_profile(reference.profile, "uint8", 255, 1)
        coverage_profile = output_profile(reference.profile, "uint8", 255, 1)

        evaluated_cells = 0
        susceptible_cells = 0
        score_sum = 0.0
        review_band_cells = 0
        unknown_class_cells = 0
        class_counts: dict[int, int] = {}

        with (
            rasterio.open(SCORE_RASTER, "w", **score_profile) as score_destination,
            rasterio.open(BINARY_RASTER, "w", **binary_profile) as binary_destination,
            rasterio.open(COVERAGE_RASTER, "w", **coverage_profile) as coverage_destination,
        ):
            for sequence, window in enumerate(windows, start=1):
                elevation = elevation_source.read(1, window=window).astype(np.float32)
                slope = slope_source.read(1, window=window).astype(np.float32)
                distance = distance_source.read(1, window=window).astype(np.float32)
                land_cover = land_cover_source.read(1, window=window)
                valid = valid_source.read(1, window=window) == 1
                valid &= np.isfinite(elevation) & np.isfinite(slope) & np.isfinite(distance)
                if elevation_source.nodata is not None:
                    valid &= elevation != elevation_source.nodata
                if slope_source.nodata is not None:
                    valid &= slope != slope_source.nodata
                if distance_source.nodata is not None:
                    valid &= distance != distance_source.nodata

                score_output = np.full(valid.shape, -9999.0, dtype=np.float32)
                binary_output = np.full(valid.shape, 255, dtype=np.uint8)
                coverage_output = np.full(valid.shape, 255, dtype=np.uint8)
                coverage_output[valid] = 1

                if np.any(valid):
                    features, recognised = construct_features(
                        elevation, slope, distance, land_cover, normalisation
                    )
                    matrix = features[valid]
                    scores = estimator.predict_proba(matrix)[:, 1].astype(np.float32)
                    predictions = scores >= threshold
                    score_output[valid] = scores
                    binary_output[valid] = predictions.astype(np.uint8)

                    evaluated_cells += int(scores.size)
                    susceptible_cells += int(np.sum(predictions))
                    score_sum += float(np.sum(scores, dtype=np.float64))
                    review_band_cells += int(np.sum((scores >= 0.002) & (scores < 0.01)))
                    unknown_class_cells += int(np.sum(valid & ~recognised))
                    codes, counts = np.unique(land_cover[valid], return_counts=True)
                    for code, count in zip(codes.tolist(), counts.tolist()):
                        class_counts[int(code)] = class_counts.get(int(code), 0) + int(count)

                score_destination.write(score_output, 1, window=window)
                binary_destination.write(binary_output, 1, window=window)
                coverage_destination.write(coverage_output, 1, window=window)
                if sequence % 50 == 0 or sequence == len(windows):
                    print(f"Processed window {sequence}/{len(windows)}")

    if evaluated_cells == 0:
        raise RuntimeError("No valid full-region cells were evaluated.")
    cell_area_km2 = abs(transform.a * transform.e) / 1_000_000
    evaluated_area = evaluated_cells * cell_area_km2
    susceptible_area = susceptible_cells * cell_area_km2
    susceptible_percentage = 100 * susceptible_cells / evaluated_cells
    mean_score = score_sum / evaluated_cells
    processing_seconds = time.perf_counter() - started

    print("Creating regional evidence figure...")
    create_figure(threshold)

    summary = {
        "experiment": "ClimateAdapt V3 frozen logistic full-region inference",
        "analysis_type": "label-blind regional susceptibility screening",
        "primary_model": "logistic regression",
        "model_tribe": "symbolist/statistical linear classifier",
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "model_retrained": False,
        "threshold_modified": False,
        "flood_target_raster_accessed": False,
        "flood_area_id_raster_accessed": False,
        "frozen_threshold": threshold,
        "threshold_source": "V3 Boulder Bank and Delaware Bay validation split",
        "input_channels": INPUT_CHANNELS,
        "normalisation_source": normalisation_source,
        "normalisation": normalisation,
        "feature_parity_check": feature_check,
        "grid_width": width,
        "grid_height": height,
        "crs": crs,
        "resolution_metres": abs(transform.a),
        "evaluated_cells": evaluated_cells,
        "evaluated_area_square_kilometres": evaluated_area,
        "susceptible_cells": susceptible_cells,
        "susceptible_area_square_kilometres": susceptible_area,
        "susceptible_percentage": susceptible_percentage,
        "mean_susceptibility_score": mean_score,
        "near_threshold_review_band_0_002_to_0_01_cells": review_band_cells,
        "unknown_land_cover_cells_mapped_to_group_0": unknown_class_cells,
        "land_cover_class_counts": class_counts,
        "processing_time_seconds": processing_seconds,
        "interpretation_warning": (
            "The score is an uncalibrated model susceptibility indicator, not a "
            "literal flood probability, hydrodynamic depth estimate, emergency "
            "warning or regulatory determination."
        ),
        "artifacts": {
            "score_raster": str(SCORE_RASTER.relative_to(PROJECT_ROOT)),
            "binary_raster": str(BINARY_RASTER.relative_to(PROJECT_ROOT)),
            "coverage_raster": str(COVERAGE_RASTER.relative_to(PROJECT_ROOT)),
            "figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    scalar_rows = []
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalar_rows.append({"metric": key, "value": value})
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(scalar_rows)

    print("\n" + "=" * 78)
    print("V3 FULL-REGION LOGISTIC INFERENCE COMPLETED")
    print("=" * 78)
    print(f"Evaluated cells:          {evaluated_cells:,}")
    print(f"Evaluated land area:      {evaluated_area:,.3f} km²")
    print(f"Susceptible cells:        {susceptible_cells:,}")
    print(f"Susceptible area:         {susceptible_area:,.3f} km²")
    print(f"Susceptible percentage:   {susceptible_percentage:.2f}%")
    print(f"Mean susceptibility score:{mean_score:.6f}")
    print(f"Unknown LCDB cells:       {unknown_class_cells:,}")
    print("-" * 78)
    print(f"Score raster:    {SCORE_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Binary raster:   {BINARY_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Coverage raster: {COVERAGE_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Summary JSON:    {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Summary CSV:     {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Evidence figure: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Processing time: {processing_seconds:.2f} seconds")
    print("=" * 78)
    print("The susceptibility score is not a calibrated flood probability.")
    print("No labels, model weights or thresholds were changed.")


if __name__ == "__main__":
    main()
