"""Explainability and environmental data-bias audit for ClimateAdapt U-Net V2.

This script performs post-hoc evaluation only. It never changes the trained
weights or the Rocks Road operational threshold. Grouped input ablation is
used to estimate model reliance on elevation, slope, coast distance and land
cover. Performance is then described across land-cover and terrain subgroups,
and representation is compared across the geographic splits.

The word "bias" here refers to spatial/environmental dataset bias. This is not
a demographic fairness audit and the ablation results are not causal effects.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climateadapt.models.unet_v2 import CompactUNetV2  # noqa: E402


RASTER_DIR = ROOT / "data" / "processed" / "combined_rasters"
TILE_DIR = ROOT / "data" / "processed" / "tiles"
MANIFEST_PATH = TILE_DIR / "tile_manifest.csv"
REFERENCE_PATH = RASTER_DIR / "elevation_5m.tif"
ELEVATION_PATH = RASTER_DIR / "elevation_5m.tif"
SLOPE_PATH = RASTER_DIR / "slope_degrees_5m.tif"
LAND_COVER_PATH = RASTER_DIR / "land_cover_class_5m.tif"
DISTANCE_PATH = RASTER_DIR / "distance_to_coast_5m.tif"
FLOOD_PATH = RASTER_DIR / "flood_target_5m.tif"
VALID_PATH = RASTER_DIR / "valid_land_mask_5m.tif"
SPLIT_PATH = RASTER_DIR / "split_mask_5m.tif"
CHECKPOINT_PATH = ROOT / "models" / "climateadapt_unet_v2_best.pt"
THRESHOLD_PATH = ROOT / "outputs" / "validation_v2" / "selected_threshold_v2.json"
FULL_REGION_SUMMARY = (
    ROOT
    / "outputs"
    / "full_region_inference_v2"
    / "nelson_full_region_summary_v2.json"
)

OUTPUT_DIR = ROOT / "outputs" / "explainability_bias_v2"
FIGURE_DIR = ROOT / "evidence" / "figures"
ABLATION_CSV = OUTPUT_DIR / "feature_group_ablation_v2.csv"
SUBGROUP_CSV = OUTPUT_DIR / "environmental_subgroup_metrics_v2.csv"
SPLIT_CSV = OUTPUT_DIR / "geographic_split_representation_v2.csv"
SUMMARY_JSON = OUTPUT_DIR / "explainability_bias_audit_v2.json"
FIGURE_PATH = FIGURE_DIR / "explainability_bias_audit_v2.png"

SPLIT_NAMES = {1: "Training", 2: "Validation", 3: "Test", 4: "Demonstration"}
MIN_SUBGROUP_PIXELS = 250


def first_present(mapping: dict, names: list[str], default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def find_threshold(value: object) -> float | None:
    preferred = (
        "operational_threshold",
        "safety_threshold",
        "selected_threshold",
        "threshold",
    )
    if isinstance(value, dict):
        for wanted in preferred:
            for key, nested in value.items():
                if key.lower() == wanted and isinstance(nested, (int, float)):
                    return float(nested)
        for nested in value.values():
            result = find_threshold(nested)
            if result is not None:
                return result
    if isinstance(value, list):
        for nested in value:
            result = find_threshold(nested)
            if result is not None:
                return result
    return None


def require_files(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))


def resolve_tile_path(value: str) -> Path:
    relative = str(value).replace("\\", "/")
    for candidate in (TILE_DIR / relative, ROOT / relative, Path(relative)):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate tile: {value}")


def load_tile(path: Path, channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as tile:
        feature_key = "features" if "features" in tile.files else "image"
        image = np.asarray(tile[feature_key], dtype=np.float32)
        target = np.squeeze(np.asarray(tile["target"], dtype=np.uint8))
        valid = np.squeeze(np.asarray(tile["valid_mask"], dtype=np.uint8)).astype(bool)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D features in {path}; received {image.shape}")
    if image.shape[0] != channels and image.shape[-1] == channels:
        image = np.moveaxis(image, -1, 0)
    if image.shape[0] != channels:
        raise ValueError(f"{path.name} has {image.shape[0]} channels; expected {channels}")
    if image.shape[1:] != target.shape or target.shape != valid.shape:
        raise ValueError(f"Spatial mismatch in {path.name}")
    return image, target, valid


def clean_state_dict(state_dict: dict) -> dict:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def raised_cosine(height: int, width: int) -> np.ndarray:
    """Positive overlap weights; the floor preserves coverage at outer edges."""
    row = np.hanning(height) if height > 2 else np.ones(height)
    column = np.hanning(width) if width > 2 else np.ones(width)
    weight = np.outer(row, column).astype(np.float32)
    return np.maximum(weight, 1e-3)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(truth: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    truth_bool = truth.astype(bool)
    prediction = probability >= threshold
    tp = int(np.sum(prediction & truth_bool))
    tn = int(np.sum(~prediction & ~truth_bool))
    fp = int(np.sum(prediction & ~truth_bool))
    fn = int(np.sum(~prediction & truth_bool))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    f2 = safe_divide(5 * precision * recall, 4 * precision + recall)
    result = {
        "pixels": int(truth.size),
        "positive_pixels": int(truth_bool.sum()),
        "flood_prevalence": float(np.mean(truth_bool)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "f2_score": f2,
        "dice": dice,
        "iou": safe_divide(tp, tp + fp + fn),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }
    if np.unique(truth).size == 2:
        result["roc_auc"] = float(roc_auc_score(truth, probability))
        result["average_precision"] = float(average_precision_score(truth, probability))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read(1)


def main() -> None:
    required = [
        MANIFEST_PATH,
        REFERENCE_PATH,
        ELEVATION_PATH,
        SLOPE_PATH,
        LAND_COVER_PATH,
        DISTANCE_PATH,
        FLOOD_PATH,
        VALID_PATH,
        SPLIT_PATH,
        CHECKPOINT_PATH,
        THRESHOLD_PATH,
    ]
    require_files(required)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    threshold_payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    threshold = find_threshold(threshold_payload)
    if threshold is None or not 0.0 < threshold < 1.0:
        raise ValueError("A valid frozen operational threshold was not found.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    configuration = checkpoint.get("configuration", {})
    channels = int(
        first_present(
            checkpoint,
            ["input_channels", "in_channels"],
            first_present(configuration, ["input_channels", "in_channels"], 25),
        )
    )
    base_channels = int(
        first_present(
            checkpoint,
            ["base_channels"],
            first_present(configuration, ["base_channels"], 16),
        )
    )
    dropout = float(
        first_present(
            checkpoint,
            ["dropout_probability", "dropout"],
            first_present(configuration, ["dropout_probability", "dropout"], 0.2),
        )
    )
    epoch = int(first_present(checkpoint, ["epoch"], -1))
    state_dict = first_present(
        checkpoint, ["model_state_dict", "state_dict", "model"], checkpoint
    )
    if not isinstance(state_dict, dict):
        raise TypeError("Could not find a model state dictionary in the checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactUNetV2(
        input_channels=channels,
        base_channels=base_channels,
        dropout_probability=dropout,
    ).to(device)
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()

    manifest = pd.read_csv(MANIFEST_PATH)
    split_text = manifest["split"].astype(str).str.lower().str.strip()
    test_manifest = manifest.loc[split_text == "test"].copy()
    if test_manifest.empty:
        raise ValueError("No independent test tiles were found.")

    with rasterio.open(REFERENCE_PATH) as reference:
        height, width = reference.height, reference.width

    records: list[dict] = []
    target_canvas = np.zeros((height, width), dtype=np.uint8)
    valid_canvas = np.zeros((height, width), dtype=bool)
    for _, row in test_manifest.iterrows():
        image, target, valid = load_tile(resolve_tile_path(row["file"]), channels)
        r0, c0 = int(row["row"]), int(row["column"])
        r1, c1 = min(r0 + target.shape[0], height), min(c0 + target.shape[1], width)
        local_h, local_w = r1 - r0, c1 - c0
        if local_h <= 0 or local_w <= 0:
            continue
        image = image[:, :local_h, :local_w]
        target = target[:local_h, :local_w]
        valid = valid[:local_h, :local_w]
        destination = np.s_[r0:r1, c0:c1]
        target_canvas[destination] = np.maximum(
            target_canvas[destination], target * valid.astype(np.uint8)
        )
        valid_canvas[destination] |= valid
        records.append(
            {
                "image": image,
                "valid": valid,
                "destination": destination,
                "weight": raised_cosine(local_h, local_w),
            }
        )
    if not records:
        raise RuntimeError("No usable test tiles were loaded.")

    print("\n" + "=" * 78)
    print("CLIMATEADAPT EXPERIMENT 2 – EXPLAINABILITY AND DATA-BIAS AUDIT")
    print("=" * 78)
    print("Method: grouped feature ablation plus environmental subgroup audit")
    print("Area: independent Tahunanui test region")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Frozen operational threshold: {threshold:.2f}")
    print(f"Device: {device}")
    print(f"Input channels: {channels}")
    print(f"Test tiles: {len(records)}")
    print("No model weights or thresholds will be changed.\n")

    def infer(ablate: list[int] | None) -> np.ndarray:
        probability_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        with torch.inference_mode():
            for record in records:
                image = record["image"].copy()
                if ablate:
                    image[ablate, :, :] = 0.0
                tensor = torch.from_numpy(image[None]).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    probability = torch.sigmoid(model(tensor))
                probability = probability.squeeze().float().cpu().numpy()
                valid = record["valid"]
                weight = record["weight"] * valid
                destination = record["destination"]
                probability_sum[destination] += probability * weight
                weight_sum[destination] += weight
        if np.any(valid_canvas & (weight_sum <= 0)):
            raise RuntimeError("Inference failed to cover every test pixel.")
        canvas = np.full((height, width), np.nan, dtype=np.float32)
        canvas[valid_canvas] = probability_sum[valid_canvas] / weight_sum[valid_canvas]
        return canvas

    groups = {
        "Baseline – all inputs": None,
        "Elevation removed": [0],
        "Slope removed": [1],
        "Coast distance removed": [2],
        "Land cover removed": list(range(3, channels)),
    }
    truth = target_canvas[valid_canvas].astype(np.uint8)
    ablation_rows = []
    baseline_probability = None
    baseline_metrics = None
    for label, channel_group in groups.items():
        print(f"Running: {label}")
        canvas = infer(channel_group)
        probability = canvas[valid_canvas].astype(np.float64)
        metrics = binary_metrics(truth, probability, threshold)
        if baseline_probability is None:
            baseline_probability = probability.copy()
            baseline_metrics = metrics.copy()
        row = {
            "feature_group": label,
            "channels_removed": "none" if channel_group is None else ",".join(map(str, channel_group)),
            **metrics,
            "mean_absolute_probability_change": float(
                np.mean(np.abs(probability - baseline_probability))
            ),
            "delta_average_precision": float(
                (metrics["average_precision"] or 0.0)
                - (baseline_metrics["average_precision"] or 0.0)
            ),
            "delta_dice": float(metrics["dice"] - baseline_metrics["dice"]),
        }
        ablation_rows.append(row)

    ablation_frame = pd.DataFrame(ablation_rows)
    ablation_frame.to_csv(ABLATION_CSV, index=False)
    assert baseline_probability is not None and baseline_metrics is not None

    # Read aligned environmental variables for descriptive subgroup evaluation.
    elevation = read_band(ELEVATION_PATH)[valid_canvas]
    slope = read_band(SLOPE_PATH)[valid_canvas]
    distance = read_band(DISTANCE_PATH)[valid_canvas]
    land_cover = read_band(LAND_COVER_PATH)[valid_canvas].astype(int)

    subgroup_rows: list[dict] = []

    def add_subgroup(group_type: str, name: str, selected: np.ndarray) -> None:
        count = int(selected.sum())
        if count < MIN_SUBGROUP_PIXELS:
            return
        metrics = binary_metrics(
            truth[selected], baseline_probability[selected], threshold
        )
        subgroup_rows.append(
            {
                "group_type": group_type,
                "group": name,
                **metrics,
                "mean_probability": float(np.mean(baseline_probability[selected])),
            }
        )

    for code in sorted(np.unique(land_cover)):
        add_subgroup("land_cover", f"LCDB {code}", land_cover == code)

    elevation_groups = [
        ("Elevation <= 2 m", elevation <= 2),
        ("Elevation 2–5 m", (elevation > 2) & (elevation <= 5)),
        ("Elevation 5–10 m", (elevation > 5) & (elevation <= 10)),
        ("Elevation > 10 m", elevation > 10),
    ]
    distance_groups = [
        ("Coast distance <= 100 m", distance <= 100),
        ("Coast distance 100–500 m", (distance > 100) & (distance <= 500)),
        ("Coast distance 500–1000 m", (distance > 500) & (distance <= 1000)),
        ("Coast distance > 1000 m", distance > 1000),
    ]
    slope_groups = [
        ("Slope <= 5°", slope <= 5),
        ("Slope 5–15°", (slope > 5) & (slope <= 15)),
        ("Slope > 15°", slope > 15),
    ]
    for name, selected in elevation_groups:
        add_subgroup("elevation_band", name, selected)
    for name, selected in distance_groups:
        add_subgroup("coast_distance_band", name, selected)
    for name, selected in slope_groups:
        add_subgroup("slope_band", name, selected)

    subgroup_frame = pd.DataFrame(subgroup_rows)
    subgroup_frame.to_csv(SUBGROUP_CSV, index=False)

    flood = read_band(FLOOD_PATH)
    valid = read_band(VALID_PATH) == 1
    split = read_band(SPLIT_PATH)
    split_rows = []
    for split_id, split_name in SPLIT_NAMES.items():
        selected = valid & (split == split_id)
        cells = int(selected.sum())
        positives = int(np.sum((flood == 1) & selected))
        split_rows.append(
            {
                "split_id": split_id,
                "split": split_name,
                "cells": cells,
                "flood_cells": positives,
                "non_flood_cells": cells - positives,
                "flood_percentage": 100.0 * positives / cells if cells else 0.0,
            }
        )
    split_frame = pd.DataFrame(split_rows)
    split_frame.to_csv(SPLIT_CSV, index=False)

    unseen_classes: list[int] = []
    unseen_cells = 0
    if FULL_REGION_SUMMARY.exists():
        full_summary = json.loads(FULL_REGION_SUMMARY.read_text(encoding="utf-8"))
        unseen_classes = [int(x) for x in full_summary.get("unseen_land_cover_classes", [])]
        unseen_cells = int(full_summary.get("unseen_land_cover_cells", 0))

    eligible = subgroup_frame[
        subgroup_frame["positive_pixels"].between(1, subgroup_frame["pixels"] - 1)
    ]
    recall_range = (
        float(eligible["recall"].max() - eligible["recall"].min())
        if not eligible.empty
        else 0.0
    )
    prevalence_range = float(
        split_frame["flood_percentage"].max() - split_frame["flood_percentage"].min()
    )
    ablated = ablation_frame.iloc[1:].copy()
    most_influential = ablated.sort_values("delta_average_precision").iloc[0]
    warnings = []
    if prevalence_range >= 10:
        warnings.append(
            "Flood prevalence differs materially between geographic splits; aggregate metrics should not be treated as location-invariant."
        )
    if recall_range >= 0.20:
        warnings.append(
            "Recall varies materially across environmental subgroups; human review and subgroup reporting are required."
        )
    if unseen_classes:
        warnings.append(
            "The full-region screen contains LCDB classes unseen during training; their outputs require additional caution."
        )

    summary = {
        "experiment": "Experiment 2 – GroupNorm Compact U-Net V2",
        "audit_type": "post-hoc grouped ablation and environmental/spatial data-bias audit",
        "demographic_fairness_audit": False,
        "causal_explanation": False,
        "model_retrained": False,
        "threshold_modified": False,
        "test_area": "Tahunanui",
        "threshold_source": "Rocks Road validation region",
        "operational_threshold": threshold,
        "checkpoint_epoch": epoch,
        "input_channels": channels,
        "evaluated_unique_test_pixels": int(valid_canvas.sum()),
        "baseline_metrics": baseline_metrics,
        "most_influential_ablation_by_average_precision_drop": {
            "feature_group": str(most_influential["feature_group"]),
            "delta_average_precision": float(most_influential["delta_average_precision"]),
            "delta_dice": float(most_influential["delta_dice"]),
        },
        "geographic_split_flood_prevalence_range_percentage_points": prevalence_range,
        "environmental_subgroup_recall_range": recall_range,
        "full_region_unseen_land_cover_classes": unseen_classes,
        "full_region_unseen_land_cover_cells": unseen_cells,
        "warnings": warnings,
        "required_controls": [
            "Retain the frozen validation threshold and independent test protocol.",
            "Report subgroup support and flood prevalence beside performance metrics.",
            "Flag unseen land-cover classes and high-uncertainty outputs for human review.",
            "Do not interpret ablation importance as physical causation.",
            "Do not use this screening model as a hydrodynamic or emergency-warning product.",
        ],
        "outputs": {
            "ablation_csv": str(ABLATION_CSV.relative_to(ROOT)),
            "subgroup_csv": str(SUBGROUP_CSV.relative_to(ROOT)),
            "split_csv": str(SPLIT_CSV.relative_to(ROOT)),
            "figure": str(FIGURE_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    ablation_plot = ablation_frame.iloc[1:].sort_values("delta_average_precision")
    axes[0, 0].barh(
        ablation_plot["feature_group"].str.replace(" removed", "", regex=False),
        -ablation_plot["delta_average_precision"],
        color="#3465a4",
    )
    axes[0, 0].axvline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Feature reliance: drop in average precision")
    axes[0, 0].set_xlabel("Baseline AP − ablated AP (larger = stronger reliance)")

    axes[0, 1].bar(
        split_frame["split"], split_frame["flood_percentage"], color="#e69f00"
    )
    axes[0, 1].set_title("Geographic split representation")
    axes[0, 1].set_ylabel("Flood pixels (%)")
    axes[0, 1].tick_params(axis="x", rotation=20)

    land_plot = subgroup_frame[
        (subgroup_frame["group_type"] == "land_cover")
        & subgroup_frame["roc_auc"].notna()
    ].sort_values("pixels", ascending=False).head(10)
    x = np.arange(len(land_plot))
    axes[1, 0].bar(x - 0.18, land_plot["precision"], width=0.36, label="Precision")
    axes[1, 0].bar(x + 0.18, land_plot["recall"], width=0.36, label="Recall")
    axes[1, 0].set_xticks(x, land_plot["group"], rotation=45, ha="right")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Largest evaluable LCDB subgroups")
    axes[1, 0].legend()

    environmental = subgroup_frame[
        subgroup_frame["group_type"].isin(
            ["elevation_band", "coast_distance_band", "slope_band"]
        )
        & subgroup_frame["roc_auc"].notna()
    ]
    axes[1, 1].scatter(
        environmental["flood_prevalence"] * 100,
        environmental["recall"],
        s=np.clip(environmental["pixels"] / 20, 25, 500),
        c=environmental["balanced_accuracy"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1, 1].set_title("Environmental subgroup performance")
    axes[1, 1].set_xlabel("Flood prevalence (%)")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_ylim(0, 1.05)

    figure.suptitle(
        "ClimateAdapt AI – Explainability and Environmental Data-Bias Audit",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Post-hoc test evaluation only. Importance is descriptive, not causal; subgroup results require adequate support.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("EXPLAINABILITY AND DATA-BIAS AUDIT COMPLETED")
    print("=" * 78)
    print(f"Unique test pixels:          {int(valid_canvas.sum()):,}")
    print(f"Baseline average precision: {baseline_metrics['average_precision']:.4f}")
    print(f"Baseline Dice:              {baseline_metrics['dice']:.4f}")
    print(f"Strongest AP ablation:      {most_influential['feature_group']}")
    print(f"AP change:                  {most_influential['delta_average_precision']:+.4f}")
    print(f"Split prevalence range:     {prevalence_range:.2f} percentage points")
    print(f"Subgroup recall range:      {recall_range:.4f}")
    print(f"Unseen full-region classes: {unseen_classes or 'none reported'}")
    print(f"Warnings recorded:          {len(warnings)}")
    print("-" * 78)
    print(f"Ablation CSV:  {ABLATION_CSV.relative_to(ROOT)}")
    print(f"Subgroup CSV:  {SUBGROUP_CSV.relative_to(ROOT)}")
    print(f"Split CSV:     {SPLIT_CSV.relative_to(ROOT)}")
    print(f"Summary JSON:  {SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Figure:        {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print("=" * 78)
    print("No model weights or thresholds were changed.")
    print("This is an environmental/spatial data-bias audit, not demographic fairness.")


if __name__ == "__main__":
    main()
