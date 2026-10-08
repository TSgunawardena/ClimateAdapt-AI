"""Apply a Physicalist terrain/connectivity guardrail to Experiment 2.

This post-processing stage does not retrain the U-Net and does not use the
Tāhunanui test labels. It learns broad physical limits from flood pixels in
the training split only, then checks the MC-dropout mean susceptibility map.

Outputs distinguish:
  * physically supported susceptible cells;
  * terrain-rule rejections;
  * predictions disconnected from the coast; and
  * supported predictions requiring human review because uncertainty is high.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]

RASTER_DIR = ROOT / "data" / "processed" / "combined_rasters"
UNCERTAINTY_DIR = ROOT / "outputs" / "uncertainty_v2"
VALIDATION_JSON = ROOT / "outputs" / "validation_v2" / "selected_threshold_v2.json"

ELEVATION_PATH = RASTER_DIR / "elevation_5m.tif"
SLOPE_PATH = RASTER_DIR / "slope_degrees_5m.tif"
DISTANCE_PATH = RASTER_DIR / "distance_to_coast_5m.tif"
FLOOD_TARGET_PATH = RASTER_DIR / "flood_target_5m.tif"
VALID_MASK_PATH = RASTER_DIR / "valid_land_mask_5m.tif"
SPLIT_MASK_PATH = RASTER_DIR / "split_mask_5m.tif"

MEAN_PROBABILITY_PATH = UNCERTAINTY_DIR / "tahunanui_mc_mean_probability_v2.tif"
EPISTEMIC_STD_PATH = UNCERTAINTY_DIR / "tahunanui_mc_epistemic_std_v2.tif"
ENTROPY_PATH = UNCERTAINTY_DIR / "tahunanui_mc_predictive_entropy_v2.tif"

OUTPUT_DIR = ROOT / "outputs" / "physicalist_guardrail_v2"
FIGURE_DIR = ROOT / "evidence" / "figures"

# Split IDs created by script 08.
TRAIN_SPLIT_ID = 1
TEST_SPLIT_ID = 3

# Broad limits learned from training flood cells. A high percentile prevents
# isolated training outliers from defining unlimited physical plausibility.
TRAINING_LIMIT_QUANTILE = 0.995

# Small safety margins make the rule a guardrail rather than a rigid simulator.
ELEVATION_MARGIN_METRES = 0.50
SLOPE_MARGIN_DEGREES = 1.00
COAST_DISTANCE_MARGIN_METRES = 50.0

# Connectivity permits gaps of up to roughly 10 m in the 5 m raster.
CONNECTIVITY_BRIDGE_PIXELS = 2
COASTAL_SEED_DISTANCE_METRES = 100.0

# Review rather than suppress predictions in the most uncertain 10%.
UNCERTAINTY_REVIEW_QUANTILE = 0.90
ENTROPY_REVIEW_THRESHOLD = 0.90


def require_files(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))


def read_raster(path: Path) -> tuple[np.ndarray, dict, float | int | None]:
    with rasterio.open(path) as source:
        return source.read(1), source.profile.copy(), source.nodata


def assert_aligned(reference_path: Path, other_paths: list[Path]) -> None:
    with rasterio.open(reference_path) as reference:
        signature = (reference.width, reference.height, reference.crs, reference.transform)
        for path in other_paths:
            with rasterio.open(path) as source:
                candidate = (source.width, source.height, source.crs, source.transform)
                if candidate != signature:
                    raise ValueError(
                        f"Raster is not aligned with {reference_path.name}: {path.name}"
                    )


def find_numeric_value(value: object, preferred_keys: tuple[str, ...]) -> float | None:
    """Recursively locate a numeric threshold while tolerating JSON variants."""
    if isinstance(value, dict):
        for preferred in preferred_keys:
            for key, nested in value.items():
                if key.lower() == preferred.lower() and isinstance(nested, (int, float)):
                    return float(nested)
        for nested in value.values():
            result = find_numeric_value(nested, preferred_keys)
            if result is not None:
                return result
    elif isinstance(value, list):
        for nested in value:
            result = find_numeric_value(nested, preferred_keys)
            if result is not None:
                return result
    return None


def load_operational_threshold(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold = find_numeric_value(
        payload,
        (
            "operational_threshold",
            "safety_threshold",
            "selected_threshold",
            "threshold",
        ),
    )
    if threshold is None or not 0.0 < threshold < 1.0:
        raise ValueError(f"Could not find a valid operational threshold in {path}")
    return threshold


def percentile_limit(values: np.ndarray, margin: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite training flood values were available.")
    return float(np.quantile(finite, TRAINING_LIMIT_QUANTILE) + margin)


def safe_percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def save_raster(
    output_path: Path,
    data: np.ndarray,
    reference_profile: dict,
    dtype: str,
    nodata: float | int,
) -> None:
    profile = reference_profile.copy()
    profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        predictor=2 if np.issubdtype(np.dtype(dtype), np.integer) else 3,
        tiled=False,
    )
    for key in ("blockxsize", "blockysize", "interleave"):
        profile.pop(key, None)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(data.astype(dtype), 1)


def main() -> None:
    required = [
        ELEVATION_PATH,
        SLOPE_PATH,
        DISTANCE_PATH,
        FLOOD_TARGET_PATH,
        VALID_MASK_PATH,
        SPLIT_MASK_PATH,
        MEAN_PROBABILITY_PATH,
        EPISTEMIC_STD_PATH,
        ENTROPY_PATH,
        VALIDATION_JSON,
    ]
    require_files(required)
    assert_aligned(ELEVATION_PATH, required[1:-1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("CLIMATEADAPT EXPERIMENT 2 – PHYSICALIST TERRAIN GUARDRAIL")
    print("=" * 78)
    print("Mode: post-model physical plausibility and human-review screening")
    print("Test labels will not be used to assess or tune the guardrail.")

    elevation, reference_profile, elevation_nodata = read_raster(ELEVATION_PATH)
    slope, _, slope_nodata = read_raster(SLOPE_PATH)
    distance, _, distance_nodata = read_raster(DISTANCE_PATH)
    flood_target, _, _ = read_raster(FLOOD_TARGET_PATH)
    valid_land, _, _ = read_raster(VALID_MASK_PATH)
    split_mask, _, _ = read_raster(SPLIT_MASK_PATH)
    mean_probability, _, probability_nodata = read_raster(MEAN_PROBABILITY_PATH)
    epistemic_std, _, std_nodata = read_raster(EPISTEMIC_STD_PATH)
    predictive_entropy, _, entropy_nodata = read_raster(ENTROPY_PATH)

    finite_inputs = np.isfinite(elevation) & np.isfinite(slope) & np.isfinite(distance)
    if elevation_nodata is not None:
        finite_inputs &= elevation != elevation_nodata
    if slope_nodata is not None:
        finite_inputs &= slope != slope_nodata
    if distance_nodata is not None:
        finite_inputs &= distance != distance_nodata

    probability_available = np.isfinite(mean_probability)
    if probability_nodata is not None:
        probability_available &= mean_probability != probability_nodata

    uncertainty_available = np.isfinite(epistemic_std) & np.isfinite(predictive_entropy)
    if std_nodata is not None:
        uncertainty_available &= epistemic_std != std_nodata
    if entropy_nodata is not None:
        uncertainty_available &= predictive_entropy != entropy_nodata

    training_flood = (
        (split_mask == TRAIN_SPLIT_ID)
        & (valid_land == 1)
        & (flood_target == 1)
        & finite_inputs
    )
    training_flood_count = int(training_flood.sum())
    if training_flood_count < 100:
        raise ValueError(
            f"Only {training_flood_count} valid training flood cells were found; "
            "that is insufficient to learn stable physical limits."
        )

    elevation_limit = percentile_limit(
        elevation[training_flood], ELEVATION_MARGIN_METRES
    )
    slope_limit = percentile_limit(slope[training_flood], SLOPE_MARGIN_DEGREES)
    coast_distance_limit = percentile_limit(
        distance[training_flood], COAST_DISTANCE_MARGIN_METRES
    )

    operational_threshold = load_operational_threshold(VALIDATION_JSON)
    evaluated = (
        (split_mask == TEST_SPLIT_ID)
        & (valid_land == 1)
        & finite_inputs
        & probability_available
    )
    candidate = evaluated & (mean_probability >= operational_threshold)

    terrain_ok = (
        (elevation <= elevation_limit)
        & (slope <= slope_limit)
        & (distance <= coast_distance_limit)
    )
    terrain_candidate = candidate & terrain_ok
    terrain_rejected = candidate & ~terrain_ok

    # Label a slightly expanded surface so tiny raster gaps do not break an
    # otherwise continuous coastal prediction. Original candidate cells alone
    # are retained in the final result.
    expanded_surface = ndimage.binary_dilation(
        terrain_candidate,
        structure=np.ones((3, 3), dtype=bool),
        iterations=CONNECTIVITY_BRIDGE_PIXELS,
        mask=evaluated,
    )
    component_labels, component_count = ndimage.label(
        expanded_surface, structure=np.ones((3, 3), dtype=np.uint8)
    )
    coastal_seed = evaluated & (distance <= COASTAL_SEED_DISTANCE_METRES)
    connected_labels = np.unique(component_labels[coastal_seed & expanded_surface])
    connected_labels = connected_labels[connected_labels != 0]
    connected_surface = np.isin(component_labels, connected_labels)

    supported = terrain_candidate & connected_surface
    disconnected = terrain_candidate & ~connected_surface

    uncertainty_values = epistemic_std[evaluated & uncertainty_available]
    if uncertainty_values.size == 0:
        raise ValueError("No valid uncertainty values were found in the test region.")
    uncertainty_review_limit = float(
        np.quantile(uncertainty_values, UNCERTAINTY_REVIEW_QUANTILE)
    )
    human_review = supported & uncertainty_available & (
        (epistemic_std >= uncertainty_review_limit)
        | (predictive_entropy >= ENTROPY_REVIEW_THRESHOLD)
    )
    auto_supported = supported & ~human_review

    # Codes: 0 background/non-candidate, 1 accepted, 2 review, 3 terrain
    # rejection, 4 coastal-disconnection rejection, 255 outside evaluation.
    decision = np.full(mean_probability.shape, 255, dtype=np.uint8)
    decision[evaluated] = 0
    decision[auto_supported] = 1
    decision[human_review] = 2
    decision[terrain_rejected] = 3
    decision[disconnected] = 4

    guarded_binary = np.full(mean_probability.shape, 255, dtype=np.uint8)
    guarded_binary[evaluated] = 0
    guarded_binary[supported] = 1

    review_binary = np.full(mean_probability.shape, 255, dtype=np.uint8)
    review_binary[evaluated] = 0
    review_binary[human_review] = 1

    candidate_count = int(candidate.sum())
    supported_count = int(supported.sum())
    review_count = int(human_review.sum())
    terrain_rejected_count = int(terrain_rejected.sum())
    disconnected_count = int(disconnected.sum())
    pixel_area_m2 = abs(reference_profile["transform"].a * reference_profile["transform"].e)

    summary = {
        "method": "physicalist terrain and coastal-connectivity guardrail",
        "label_blind_test_application": True,
        "operational_threshold": operational_threshold,
        "training_limit_quantile": TRAINING_LIMIT_QUANTILE,
        "training_flood_cells_used": training_flood_count,
        "physical_limits": {
            "maximum_elevation_metres": elevation_limit,
            "maximum_slope_degrees": slope_limit,
            "maximum_distance_to_coast_metres": coast_distance_limit,
            "coastal_seed_distance_metres": COASTAL_SEED_DISTANCE_METRES,
            "connectivity_bridge_pixels": CONNECTIVITY_BRIDGE_PIXELS,
        },
        "uncertainty_review": {
            "epistemic_quantile": UNCERTAINTY_REVIEW_QUANTILE,
            "epistemic_std_threshold": uncertainty_review_limit,
            "predictive_entropy_threshold": ENTROPY_REVIEW_THRESHOLD,
        },
        "results": {
            "evaluated_cells": int(evaluated.sum()),
            "original_candidate_cells": candidate_count,
            "physically_supported_cells": supported_count,
            "automatic_supported_cells": int(auto_supported.sum()),
            "human_review_cells": review_count,
            "terrain_rejected_cells": terrain_rejected_count,
            "coast_disconnected_cells": disconnected_count,
            "supported_percentage_of_candidates": safe_percentage(
                supported_count, candidate_count
            ),
            "human_review_percentage_of_supported": safe_percentage(
                review_count, supported_count
            ),
            "supported_area_km2": supported_count * pixel_area_m2 / 1_000_000.0,
            "connected_components_examined": int(component_count),
            "coast_connected_components": int(connected_labels.size),
        },
        "decision_codes": {
            "0": "not susceptible at frozen model threshold",
            "1": "physically supported",
            "2": "physically supported; human review required",
            "3": "rejected by terrain limits",
            "4": "rejected because disconnected from coastal seed",
            "255": "outside evaluated test region",
        },
        "important_note": (
            "This is an explainable screening guardrail, not a hydrodynamic flood model."
        ),
    }

    summary_json = OUTPUT_DIR / "tahunanui_physicalist_guardrail_summary_v2.json"
    summary_csv = OUTPUT_DIR / "tahunanui_physicalist_guardrail_summary_v2.csv"
    decision_path = OUTPUT_DIR / "tahunanui_physicalist_decision_v2.tif"
    binary_path = OUTPUT_DIR / "tahunanui_guarded_susceptibility_v2.tif"
    review_path = OUTPUT_DIR / "tahunanui_guardrail_human_review_v2.tif"
    figure_path = FIGURE_DIR / "tahunanui_physicalist_guardrail_v2.png"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flat_results = {
        "operational_threshold": operational_threshold,
        "maximum_elevation_metres": elevation_limit,
        "maximum_slope_degrees": slope_limit,
        "maximum_distance_to_coast_metres": coast_distance_limit,
        "epistemic_std_review_threshold": uncertainty_review_limit,
        **summary["results"],
    }
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_results.keys()))
        writer.writeheader()
        writer.writerow(flat_results)

    save_raster(decision_path, decision, reference_profile, "uint8", 255)
    save_raster(binary_path, guarded_binary, reference_profile, "uint8", 255)
    save_raster(review_path, review_binary, reference_profile, "uint8", 255)

    # Crop the evidence figure to the actual inference footprint. The source
    # rasters cover a much larger Nelson grid, which would otherwise make the
    # Tāhunanui decisions too small to interpret.
    plot_rows, plot_columns = np.where(evaluated & probability_available)
    if plot_rows.size == 0:
        raise ValueError("No evaluated pixels were available for the evidence figure.")
    plot_padding = 20
    row_start = max(0, int(plot_rows.min()) - plot_padding)
    row_stop = min(mean_probability.shape[0], int(plot_rows.max()) + plot_padding + 1)
    column_start = max(0, int(plot_columns.min()) - plot_padding)
    column_stop = min(mean_probability.shape[1], int(plot_columns.max()) + plot_padding + 1)
    crop = np.s_[row_start:row_stop, column_start:column_stop]

    transform = reference_profile["transform"]
    display_extent = (
        transform.c + column_start * transform.a,
        transform.c + column_stop * transform.a,
        transform.f + row_stop * transform.e,
        transform.f + row_start * transform.e,
    )
    evaluated_crop = evaluated[crop]
    candidate_crop = candidate[crop]
    supported_crop = supported[crop]
    human_review_crop = human_review[crop]
    decision_crop = decision[crop]
    probability_crop = mean_probability[crop]

    masked_probability = np.ma.masked_where(~evaluated_crop, probability_crop)
    original_display = np.ma.masked_where(
        ~evaluated_crop, candidate_crop.astype(float)
    )
    supported_display = np.ma.masked_where(
        ~evaluated_crop, supported_crop.astype(float)
    )
    decision_display = np.ma.masked_where(decision_crop == 255, decision_crop)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    axes[0, 0].imshow(masked_probability, cmap="viridis", vmin=0, vmax=1, extent=display_extent)
    axes[0, 0].contour(
        original_display,
        levels=[0.5],
        colors=["yellow"],
        linewidths=0.8,
        extent=display_extent,
    )
    axes[0, 0].set_title(f"Original MC susceptibility (yellow ≥ {operational_threshold:.2f})")

    axes[0, 1].imshow(
        supported_display,
        cmap=ListedColormap(["#eeeeee", "#17883e"]),
        vmin=0,
        vmax=1,
        extent=display_extent,
    )
    axes[0, 1].set_title("Physically supported susceptibility")

    decision_cmap = ListedColormap(
        ["#eeeeee", "#17883e", "#ffbf00", "#d73027", "#7b3294"]
    )
    decision_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], decision_cmap.N)
    axes[1, 0].imshow(
        decision_display,
        cmap=decision_cmap,
        norm=decision_norm,
        extent=display_extent,
    )
    axes[1, 0].set_title("Decision: green accepted, amber review, red terrain, purple disconnected")

    axes[1, 1].imshow(masked_probability, cmap="Greys", vmin=0, vmax=1, extent=display_extent)
    axes[1, 1].imshow(
        np.ma.masked_where(~human_review_crop, human_review_crop),
        cmap=ListedColormap(["#ffbf00"]),
        alpha=0.9,
        extent=display_extent,
    )
    axes[1, 1].set_title("Human-review locations from MC uncertainty")

    for axis in axes.flat:
        axis.set_xlabel("Easting (NZTM2000)")
        axis.set_ylabel("Northing (NZTM2000)")
        axis.ticklabel_format(style="plain")

    fig.suptitle("ClimateAdapt AI – Physicalist Terrain and Connectivity Guardrail", fontsize=18)
    fig.text(
        0.5,
        -0.005,
        (
            f"Training-derived limits: elevation ≤ {elevation_limit:.2f} m | "
            f"slope ≤ {slope_limit:.2f}° | coast distance ≤ {coast_distance_limit:.1f} m | "
            f"supported={safe_percentage(supported_count, candidate_count):.1f}% of candidates"
        ),
        ha="center",
        fontsize=10,
    )
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Training flood cells used:       {training_flood_count:,}")
    print(f"Frozen model threshold:          {operational_threshold:.2f}")
    print(f"Maximum elevation:               {elevation_limit:.2f} m")
    print(f"Maximum slope:                   {slope_limit:.2f} degrees")
    print(f"Maximum coastline distance:      {coast_distance_limit:.1f} m")
    print(f"Epistemic review threshold:      {uncertainty_review_limit:.4f}")
    print("-" * 78)
    print(f"Original susceptible cells:      {candidate_count:,}")
    print(f"Physically supported cells:      {supported_count:,}")
    print(f"  Automatic support:             {int(auto_supported.sum()):,}")
    print(f"  Human review required:         {review_count:,}")
    print(f"Terrain-rule rejections:         {terrain_rejected_count:,}")
    print(f"Disconnected rejections:         {disconnected_count:,}")
    print(f"Supported candidate percentage:  {safe_percentage(supported_count, candidate_count):.2f}%")
    print("\n" + "=" * 78)
    print("PHYSICALIST GUARDRAIL COMPLETED")
    print("=" * 78)
    print(f"Decision raster: {decision_path.relative_to(ROOT)}")
    print(f"Guarded binary:  {binary_path.relative_to(ROOT)}")
    print(f"Review raster:   {review_path.relative_to(ROOT)}")
    print(f"Summary JSON:    {summary_json.relative_to(ROOT)}")
    print(f"Summary CSV:     {summary_csv.relative_to(ROOT)}")
    print(f"Evidence figure: {figure_path.relative_to(ROOT)}")
    print("=" * 78)
    print("No model weights or thresholds were changed.")
    print("Tāhunanui labels were not used to tune or score the guardrail.")
    print("This guardrail supports human review; it is not a hydrodynamic simulator.")


if __name__ == "__main__":
    main()
