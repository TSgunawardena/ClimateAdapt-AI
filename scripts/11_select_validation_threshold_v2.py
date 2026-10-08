"""
Select Experiment 2 thresholds using ONLY Rocks Road validation data.

Overlapping tile predictions are averaged so every geographic raster cell
is evaluated once. Tāhunanui test data and the Waimea/Monaco demonstration
holdout are never accessed.
"""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch


# =========================================================
# Project paths and configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_FOLDER = PROJECT_ROOT / "src"

if str(SRC_FOLDER) not in sys.path:
    sys.path.insert(0, str(SRC_FOLDER))

from climateadapt.models.unet_v2 import CompactUNetV2


TILE_FOLDER = PROJECT_ROOT / "data/processed/tiles"
MANIFEST_PATH = TILE_FOLDER / "tile_manifest.csv"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models/climateadapt_unet_v2_best.pt"
)
REFERENCE_RASTER_PATH = (
    PROJECT_ROOT
    / "data/processed/combined_rasters/valid_land_mask_5m.tif"
)

OUTPUT_FOLDER = PROJECT_ROOT / "outputs/validation_v2"
FIGURE_FOLDER = PROJECT_ROOT / "evidence/figures"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

THRESHOLD_RESULTS_PATH = (
    OUTPUT_FOLDER / "validation_threshold_metrics_v2.csv"
)
SELECTED_THRESHOLD_PATH = (
    OUTPUT_FOLDER / "selected_threshold_v2.json"
)
FIGURE_PATH = (
    FIGURE_FOLDER / "validation_threshold_selection_v2.png"
)

THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01),
    2,
)
MINIMUM_OPERATIONAL_SPECIFICITY = 0.80
F_BETA = 2.0


# =========================================================
# Tile and checkpoint helpers
# =========================================================
def find_tile_file(relative_file):
    supplied_path = Path(str(relative_file))

    if supplied_path.is_absolute() and supplied_path.exists():
        return supplied_path

    possible_paths = [
        TILE_FOLDER / supplied_path,
        PROJECT_ROOT / supplied_path,
    ]

    for possible_path in possible_paths:
        if possible_path.exists():
            return possible_path

    raise FileNotFoundError(
        f"Could not locate validation tile: {relative_file}"
    )


def load_tile(tile_path, expected_channels):
    with np.load(tile_path) as tile:
        features = np.asarray(
            tile["features"],
            dtype=np.float32,
        )
        target = np.asarray(
            tile["target"],
            dtype=np.float32,
        ).squeeze()
        valid_mask = np.asarray(
            tile["valid_mask"],
            dtype=np.float32,
        ).squeeze()

    if features.ndim != 3:
        raise ValueError(
            f"Expected C x H x W features in {tile_path}, "
            f"received {features.shape}."
        )

    if (
        features.shape[0] != expected_channels
        and features.shape[-1] == expected_channels
    ):
        features = np.transpose(features, (2, 0, 1))

    if features.shape[0] != expected_channels:
        raise ValueError(
            f"Tile {tile_path.name} has {features.shape[0]} channels; "
            f"the checkpoint expects {expected_channels}."
        )

    if target.shape != valid_mask.shape:
        raise ValueError(
            f"Target and valid-mask shapes differ in {tile_path}: "
            f"{target.shape} versus {valid_mask.shape}."
        )

    return features, target, valid_mask


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "model_weights",
        ]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def clean_state_dict(state_dict):
    return {
        (
            key[7:]
            if key.startswith("module.")
            else key
        ): value
        for key, value in state_dict.items()
    }


def infer_model_dimensions(state_dict):
    convolution_weights = [
        value
        for value in state_dict.values()
        if torch.is_tensor(value) and value.ndim == 4
    ]

    if not convolution_weights:
        raise RuntimeError(
            "No convolution weights were found in the checkpoint."
        )

    first_weight = convolution_weights[0]
    return int(first_weight.shape[1]), int(first_weight.shape[0])


# =========================================================
# Metrics
# =========================================================
def calculate_metrics(target, prediction):
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)

    true_positive = int((target & prediction).sum())
    true_negative = int((~target & ~prediction).sum())
    false_positive = int((~target & prediction).sum())
    false_negative = int((target & ~prediction).sum())

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )
    dice = (
        2.0 * true_positive
        / (2.0 * true_positive + false_positive + false_negative)
        if 2.0 * true_positive + false_positive + false_negative > 0
        else 0.0
    )
    iou = (
        true_positive
        / (true_positive + false_positive + false_negative)
        if true_positive + false_positive + false_negative > 0
        else 0.0
    )
    accuracy = (
        (true_positive + true_negative)
        / (
            true_positive
            + true_negative
            + false_positive
            + false_negative
        )
    )

    beta_squared = F_BETA ** 2
    f_beta = (
        (1.0 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
        if beta_squared * precision + recall > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "dice": float(dice),
        "iou": float(iou),
        "f2": float(f_beta),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


# =========================================================
# Initial checks and model loading
# =========================================================
print("\n" + "=" * 76)
print("EXPERIMENT 2 VALIDATION THRESHOLD SELECTION")
print("=" * 76)

for required_path in [
    MANIFEST_PATH,
    CHECKPOINT_PATH,
    REFERENCE_RASTER_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required file not found: {required_path}")

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {"file", "split", "row", "column"}
missing_columns = required_columns - set(manifest.columns)

if missing_columns:
    raise KeyError(
        f"Manifest is missing columns: {sorted(missing_columns)}"
    )

validation_manifest = manifest[
    manifest["split"].str.lower() == "validation"
].copy()

if validation_manifest.empty:
    raise RuntimeError("No validation tiles were found in the manifest.")

if (manifest["split"].str.lower() == "demonstration").any():
    raise RuntimeError(
        "Protected demonstration tiles were found in the manifest."
    )

with rasterio.open(REFERENCE_RASTER_PATH) as reference_raster:
    raster_height = reference_raster.height
    raster_width = reference_raster.width

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)
state_dict = clean_state_dict(
    extract_state_dict(checkpoint)
)

input_channels = int(
    checkpoint.get("input_channels", 0)
) if isinstance(checkpoint, dict) else 0
base_channels = int(
    checkpoint.get("base_channels", 0)
) if isinstance(checkpoint, dict) else 0

if input_channels <= 0 or base_channels <= 0:
    input_channels, base_channels = infer_model_dimensions(state_dict)

dropout_probability = (
    float(checkpoint.get("dropout_probability", 0.20))
    if isinstance(checkpoint, dict)
    else 0.20
)

model = CompactUNetV2(
    input_channels=input_channels,
    base_channels=base_channels,
    dropout_probability=dropout_probability,
).to(device)
model.load_state_dict(state_dict)
model.eval()

print(f"Validation tiles: {len(validation_manifest)}")
print(
    "Checkpoint: "
    f"{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}"
)
print(
    f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}"
)
print(
    "Checkpoint validation PR-AUC: "
    f"{checkpoint.get('validation_metrics', {}).get('pr_auc', float('nan')):.4f}"
)
print(f"Reference raster: {raster_width} x {raster_height}")
print(f"Device: {device}")
print(f"Input channels: {input_channels}")
print(f"Base channels: {base_channels}")


# =========================================================
# Overlap-aware validation inference
# =========================================================
probability_sum = np.zeros(
    (raster_height, raster_width),
    dtype=np.float32,
)
prediction_count = np.zeros(
    (raster_height, raster_width),
    dtype=np.uint16,
)
target_canvas = np.zeros(
    (raster_height, raster_width),
    dtype=np.uint8,
)

print("Running inference on Rocks Road validation tiles...")

with torch.inference_mode():
    for tile_number, (_, row) in enumerate(
        validation_manifest.iterrows(),
        start=1,
    ):
        tile_path = find_tile_file(row["file"])
        features, target, valid_mask = load_tile(
            tile_path,
            input_channels,
        )

        input_tensor = (
            torch.from_numpy(features)
            .unsqueeze(0)
            .to(device)
        )
        probabilities = (
            torch.sigmoid(model(input_tensor))
            .squeeze()
            .cpu()
            .numpy()
        )

        row_offset = int(row["row"])
        column_offset = int(row["column"])
        tile_height, tile_width = target.shape
        row_end = row_offset + tile_height
        column_end = column_offset + tile_width

        if row_end > raster_height or column_end > raster_width:
            raise ValueError(
                f"Tile {tile_path.name} exceeds raster bounds."
            )

        valid_pixels = valid_mask > 0.5
        probability_view = probability_sum[
            row_offset:row_end,
            column_offset:column_end,
        ]
        count_view = prediction_count[
            row_offset:row_end,
            column_offset:column_end,
        ]
        target_view = target_canvas[
            row_offset:row_end,
            column_offset:column_end,
        ]

        probability_view[valid_pixels] += probabilities[valid_pixels]
        count_view[valid_pixels] += 1
        target_view[valid_pixels] = np.maximum(
            target_view[valid_pixels],
            target[valid_pixels].astype(np.uint8),
        )

        print(
            f"\rProcessed validation tile {tile_number}/"
            f"{len(validation_manifest)}",
            end="",
        )

print()

evaluated_pixels = prediction_count > 0

if not evaluated_pixels.any():
    raise RuntimeError("No valid validation pixels were evaluated.")

averaged_probability = np.zeros_like(probability_sum)
averaged_probability[evaluated_pixels] = (
    probability_sum[evaluated_pixels]
    / prediction_count[evaluated_pixels]
)
all_probabilities = averaged_probability[evaluated_pixels]
all_targets = target_canvas[evaluated_pixels]

total_tile_predictions = int(
    prediction_count[evaluated_pixels].sum()
)
unique_validation_pixels = int(evaluated_pixels.sum())
duplicate_predictions = (
    total_tile_predictions - unique_validation_pixels
)

print(f"Unique validation pixels: {unique_validation_pixels:,}")
print(
    "Overlapping duplicate predictions merged: "
    f"{duplicate_predictions:,}"
)
print(f"Positive validation pixels: {int(all_targets.sum()):,}")
print(
    "Validation flood percentage: "
    f"{100.0 * all_targets.mean():.3f}%"
)


# =========================================================
# Evaluate candidate thresholds
# =========================================================
results = []

for threshold in THRESHOLDS:
    metrics = calculate_metrics(
        all_targets,
        all_probabilities >= threshold,
    )
    results.append(
        {
            "threshold": float(threshold),
            **metrics,
        }
    )

dice_optimal = max(
    results,
    key=lambda result: (
        result["dice"],
        result["recall"],
        result["precision"],
    ),
)

safe_candidates = [
    result
    for result in results
    if result["specificity"] >= MINIMUM_OPERATIONAL_SPECIFICITY
]

if not safe_candidates:
    raise RuntimeError(
        "No threshold satisfies the minimum specificity guard."
    )

operational_result = max(
    safe_candidates,
    key=lambda result: (
        result["f2"],
        result["recall"],
        result["dice"],
        result["precision"],
    ),
)
selected_threshold = operational_result["threshold"]


# =========================================================
# Save metrics and frozen threshold
# =========================================================
results_frame = pd.DataFrame(results)
results_frame.to_csv(
    THRESHOLD_RESULTS_PATH,
    index=False,
)

threshold_summary = {
    "experiment": "Experiment 2",
    "selected_threshold": selected_threshold,
    "operational_threshold": selected_threshold,
    "selection_dataset": "Rocks Road validation region",
    "selection_metric": (
        "F2 score with specificity >= "
        f"{MINIMUM_OPERATIONAL_SPECIFICITY:.2f}"
    ),
    "selection_reason": (
        "Recall is weighted more strongly because false-negative coastal "
        "hazard cells are more costly in screening-level planning."
    ),
    "operational_validation_metrics": operational_result,
    "dice_optimal_threshold": dice_optimal["threshold"],
    "dice_optimal_validation_metrics": dice_optimal,
    "overlap_handling": (
        "Mean probability for each unique geographic raster cell"
    ),
    "unique_validation_pixels": unique_validation_pixels,
    "duplicate_predictions_merged": duplicate_predictions,
    "checkpoint": str(
        CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
    ),
    "checkpoint_epoch": checkpoint.get("epoch"),
    "test_dataset_used": False,
    "demonstration_dataset_used": False,
}

with SELECTED_THRESHOLD_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(threshold_summary, file, indent=4)


# =========================================================
# Evidence figure
# =========================================================
figure, axis = plt.subplots(figsize=(11, 7))

axis.plot(
    results_frame["threshold"],
    results_frame["dice"],
    label="Dice",
    linewidth=2.4,
)
axis.plot(
    results_frame["threshold"],
    results_frame["f2"],
    label="F2 safety score",
    linewidth=2.4,
)
axis.plot(
    results_frame["threshold"],
    results_frame["precision"],
    label="Precision",
    linewidth=1.8,
)
axis.plot(
    results_frame["threshold"],
    results_frame["recall"],
    label="Recall",
    linewidth=1.8,
)
axis.plot(
    results_frame["threshold"],
    results_frame["specificity"],
    label="Specificity",
    linewidth=1.6,
    alpha=0.8,
)

axis.axvline(
    selected_threshold,
    color="black",
    linestyle="--",
    linewidth=1.7,
    label=f"Operational F2 threshold = {selected_threshold:.2f}",
)
axis.axvline(
    dice_optimal["threshold"],
    color="#7b2cbf",
    linestyle=":",
    linewidth=1.7,
    label=f"Dice-optimal threshold = {dice_optimal['threshold']:.2f}",
)
axis.axhline(
    MINIMUM_OPERATIONAL_SPECIFICITY,
    color="gray",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label="Minimum operational specificity",
)

axis.scatter(
    [selected_threshold],
    [operational_result["f2"]],
    color="red",
    s=70,
    zorder=5,
)
axis.set_title(
    "ClimateAdapt AI – Experiment 2 Validation Thresholds",
    fontsize=15,
    fontweight="bold",
)
axis.set_xlabel("Probability threshold")
axis.set_ylabel("Metric score")
axis.set_xlim(0.05, 0.95)
axis.set_ylim(0.0, 1.02)
axis.grid(alpha=0.25)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
plt.close(figure)


# =========================================================
# Final output
# =========================================================
print("\n" + "=" * 76)
print("EXPERIMENT 2 VALIDATION THRESHOLDS FROZEN")
print("=" * 76)
print("Operational safety threshold")
print(f"  Threshold:   {selected_threshold:.2f}")
print(f"  F2 score:    {operational_result['f2']:.4f}")
print(f"  Dice:        {operational_result['dice']:.4f}")
print(f"  Precision:   {operational_result['precision']:.4f}")
print(f"  Recall:      {operational_result['recall']:.4f}")
print(f"  Specificity: {operational_result['specificity']:.4f}")
print("-" * 76)
print("Dice-optimal comparison threshold")
print(f"  Threshold:   {dice_optimal['threshold']:.2f}")
print(f"  Dice:        {dice_optimal['dice']:.4f}")
print(f"  Precision:   {dice_optimal['precision']:.4f}")
print(f"  Recall:      {dice_optimal['recall']:.4f}")
print("-" * 76)
print(
    "Metrics: "
    f"{THRESHOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)}"
)
print(
    "Frozen threshold: "
    f"{SELECTED_THRESHOLD_PATH.relative_to(PROJECT_ROOT)}"
)
print(
    "Evidence figure: "
    f"{FIGURE_PATH.relative_to(PROJECT_ROOT)}"
)
print("=" * 76)
print("Tāhunanui test and Waimea/Monaco demonstration data were not accessed.")
