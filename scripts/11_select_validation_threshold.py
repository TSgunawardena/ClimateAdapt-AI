"""
Select the binary decision threshold using ONLY the validation dataset.

Overlapping validation-tile predictions are averaged so each geographic
cell is evaluated only once.

The independent Tahunanui test dataset is not accessed.
"""

from pathlib import Path
import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.models.unet import CompactUNet


TILES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tiles"
)

MANIFEST_PATH = TILES_DIR / "tile_manifest.csv"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "climateadapt_unet_best.pt"
)

REFERENCE_RASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_rasters"
    / "valid_land_mask_5m.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "evidence"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_RESULTS_PATH = (
    OUTPUT_DIR
    / "validation_threshold_metrics.csv"
)

SELECTED_THRESHOLD_PATH = (
    OUTPUT_DIR
    / "selected_threshold.json"
)

FIGURE_PATH = (
    FIGURE_DIR
    / "validation_threshold_selection.png"
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def find_column(fieldnames, possible_names, required=True):
    """Find a CSV column using several possible names."""

    lowercase_map = {
        name.lower(): name
        for name in fieldnames
    }

    for possible_name in possible_names:
        if possible_name.lower() in lowercase_map:
            return lowercase_map[possible_name.lower()]

    if required:
        raise KeyError(
            f"Could not locate any of these columns: "
            f"{possible_names}\n"
            f"Available columns: {fieldnames}"
        )

    return None


def find_tile_file(row, path_column):
    """Resolve the saved NPZ tile path."""

    if path_column and row.get(path_column):
        supplied_path = Path(row[path_column])

        if supplied_path.is_absolute() and supplied_path.exists():
            return supplied_path

        possible_paths = [
            PROJECT_ROOT / supplied_path,
            TILES_DIR / supplied_path,
        ]

        for possible_path in possible_paths:
            if possible_path.exists():
                return possible_path

    tile_id = None

    for name in ["tile_id", "id", "tile", "name"]:
        if name in row and row[name]:
            tile_id = row[name]
            break

    if tile_id is not None:
        matches = list(
            TILES_DIR.rglob(f"{tile_id}.npz")
        )

        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not locate the tile file for manifest row:\n"
        f"{row}"
    )


def find_npz_array(npz_file, possible_keys, required=True):
    """Find an NPZ array using several possible key names."""

    key_map = {
        key.lower(): key
        for key in npz_file.files
    }

    for possible_key in possible_keys:
        if possible_key.lower() in key_map:
            return npz_file[
                key_map[possible_key.lower()]
            ]

    if required:
        raise KeyError(
            f"Could not find any of these NPZ keys: "
            f"{possible_keys}\n"
            f"Available keys: {npz_file.files}"
        )

    return None


def load_tile(tile_path):
    """Load input channels, flood target and valid mask."""

    with np.load(tile_path) as tile:
        inputs = find_npz_array(
            tile,
            [
                "inputs",
                "input",
                "x",
                "features",
                "image",
            ],
        )

        target = find_npz_array(
            tile,
            [
                "target",
                "targets",
                "y",
                "label",
                "mask",
                "flood_target",
            ],
        )

        valid_mask = find_npz_array(
            tile,
            [
                "valid_mask",
                "valid",
                "land_mask",
            ],
            required=False,
        )

    inputs = np.asarray(
        inputs,
        dtype=np.float32,
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    ).squeeze()

    if inputs.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional input array in "
            f"{tile_path}, but received {inputs.shape}"
        )

    # Convert H x W x C to C x H x W when necessary.
    if inputs.shape[0] != 24 and inputs.shape[-1] == 24:
        inputs = np.transpose(
            inputs,
            (2, 0, 1),
        )

    if valid_mask is None:
        valid_mask = np.ones_like(
            target,
            dtype=np.float32,
        )
    else:
        valid_mask = np.asarray(
            valid_mask,
            dtype=np.float32,
        ).squeeze()

    if target.shape != valid_mask.shape:
        raise ValueError(
            f"Target and valid-mask shapes do not match "
            f"in {tile_path}: "
            f"{target.shape} and {valid_mask.shape}"
        )

    return inputs, target, valid_mask


def extract_state_dict(checkpoint):
    """Support full checkpoints and raw state dictionaries."""

    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "model_weights",
        ]:
            if (
                key in checkpoint
                and isinstance(checkpoint[key], dict)
            ):
                return checkpoint[key]

    return checkpoint


def clean_state_dict(state_dict):
    """Remove DataParallel prefixes when present."""

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    return cleaned


def infer_model_dimensions(state_dict):
    """Infer input and base channels from the first convolution."""

    convolution_weights = [
        value
        for value in state_dict.values()
        if torch.is_tensor(value)
        and value.ndim == 4
    ]

    if not convolution_weights:
        raise RuntimeError(
            "No convolution weights were found in the checkpoint."
        )

    first_weight = convolution_weights[0]

    base_channels = int(first_weight.shape[0])
    input_channels = int(first_weight.shape[1])

    return input_channels, base_channels


def calculate_metrics(target, prediction):
    """Calculate binary segmentation metrics."""

    target = target.astype(bool)
    prediction = prediction.astype(bool)

    true_positive = int(
        np.logical_and(
            target,
            prediction,
        ).sum()
    )

    true_negative = int(
        np.logical_and(
            ~target,
            ~prediction,
        ).sum()
    )

    false_positive = int(
        np.logical_and(
            ~target,
            prediction,
        ).sum()
    )

    false_negative = int(
        np.logical_and(
            target,
            ~prediction,
        ).sum()
    )

    epsilon = 1e-8

    precision = true_positive / (
        true_positive
        + false_positive
        + epsilon
    )

    recall = true_positive / (
        true_positive
        + false_negative
        + epsilon
    )

    specificity = true_negative / (
        true_negative
        + false_positive
        + epsilon
    )

    dice = (2.0 * true_positive) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    iou = true_positive / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    accuracy = (
        true_positive + true_negative
    ) / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + epsilon
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "dice": float(dice),
        "iou": float(iou),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


# ---------------------------------------------------------------------
# Initial checks
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("VALIDATION THRESHOLD SELECTION")
print("=" * 72)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

if not REFERENCE_RASTER_PATH.exists():
    raise FileNotFoundError(
        f"Reference raster not found: "
        f"{REFERENCE_RASTER_PATH}"
    )


# ---------------------------------------------------------------------
# Load validation manifest
# ---------------------------------------------------------------------

with MANIFEST_PATH.open(
    "r",
    newline="",
    encoding="utf-8",
) as file:
    reader = csv.DictReader(file)
    manifest_rows = list(reader)
    fieldnames = reader.fieldnames

if not fieldnames:
    raise RuntimeError(
        "The tile manifest does not contain column headings."
    )

split_column = find_column(
    fieldnames,
    [
        "split",
        "dataset_split",
        "set",
        "subset",
    ],
)

path_column = find_column(
    fieldnames,
    [
        "file",
        "tile_path",
        "path",
        "file_path",
        "filepath",
        "filename",
    ],
    required=False,
)

row_column = find_column(
    fieldnames,
    [
        "row",
        "row_offset",
        "row_off",
        "y",
    ],
)

column_column = find_column(
    fieldnames,
    [
        "column",
        "col",
        "column_offset",
        "col_offset",
        "col_off",
        "x",
    ],
)

validation_rows = [
    row
    for row in manifest_rows
    if row[split_column].strip().lower()
    in {
        "validation",
        "validate",
        "valid",
        "val",
    }
]

if not validation_rows:
    raise RuntimeError(
        "No validation tiles were found in the manifest."
    )

print(
    f"Validation tiles: "
    f"{len(validation_rows)}"
)

print(
    "Checkpoint: "
    f"{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}"
)


# ---------------------------------------------------------------------
# Load reference raster dimensions
# ---------------------------------------------------------------------

with rasterio.open(
    REFERENCE_RASTER_PATH
) as reference_raster:
    raster_height = reference_raster.height
    raster_width = reference_raster.width

print(
    f"Reference raster: "
    f"{raster_width} columns x "
    f"{raster_height} rows"
)


# ---------------------------------------------------------------------
# Load trained U-Net
# ---------------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

state_dict = extract_state_dict(checkpoint)
state_dict = clean_state_dict(state_dict)

input_channels, base_channels = (
    infer_model_dimensions(state_dict)
)

model = CompactUNet(
    input_channels=input_channels,
    base_channels=base_channels,
    dropout_probability=0.20,
).to(device)

model.load_state_dict(state_dict)
model.eval()

print(f"Device: {device}")
print(f"Input channels: {input_channels}")
print(f"Base channels: {base_channels}")


# ---------------------------------------------------------------------
# Prepare overlap-aware geographic canvases
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Run validation inference
# ---------------------------------------------------------------------

print(
    "Running inference on Rocks Road validation tiles..."
)

with torch.inference_mode():
    for tile_number, row in enumerate(
        validation_rows,
        start=1,
    ):
        tile_path = find_tile_file(
            row,
            path_column,
        )

        inputs, target, valid_mask = load_tile(
            tile_path
        )

        input_tensor = (
            torch.from_numpy(inputs)
            .unsqueeze(0)
            .to(device)
        )

        logits = model(input_tensor)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        probabilities = (
            torch.sigmoid(logits)
            .squeeze()
            .cpu()
            .numpy()
        )

        row_offset = int(row[row_column])
        column_offset = int(row[column_column])

        tile_height, tile_width = target.shape

        row_end = row_offset + tile_height
        column_end = column_offset + tile_width

        if (
            row_end > raster_height
            or column_end > raster_width
        ):
            raise ValueError(
                f"Tile {tile_path.name} exceeds "
                f"the reference-raster bounds."
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

        probability_view[valid_pixels] += (
            probabilities[valid_pixels]
        )

        count_view[valid_pixels] += 1

        target_view[valid_pixels] = np.maximum(
            target_view[valid_pixels],
            target[valid_pixels].astype(np.uint8),
        )

        print(
            f"\rProcessed validation tile "
            f"{tile_number}/"
            f"{len(validation_rows)}",
            end="",
        )

print()


# ---------------------------------------------------------------------
# Average overlapping predictions
# ---------------------------------------------------------------------

evaluated_pixels = prediction_count > 0

if not evaluated_pixels.any():
    raise RuntimeError(
        "No valid validation pixels were evaluated."
    )

averaged_probability = np.zeros_like(
    probability_sum,
    dtype=np.float32,
)

averaged_probability[evaluated_pixels] = (
    probability_sum[evaluated_pixels]
    / prediction_count[evaluated_pixels]
)

all_probabilities = averaged_probability[
    evaluated_pixels
].astype(np.float32)

all_targets = target_canvas[
    evaluated_pixels
].astype(np.uint8)

total_tile_predictions = int(
    prediction_count[evaluated_pixels].sum()
)

unique_validation_pixels = int(
    evaluated_pixels.sum()
)

duplicate_predictions = (
    total_tile_predictions
    - unique_validation_pixels
)

print(
    f"Unique validation pixels: "
    f"{unique_validation_pixels:,}"
)

print(
    f"Overlapping duplicate predictions merged: "
    f"{duplicate_predictions:,}"
)

print(
    f"Positive validation pixels: "
    f"{all_targets.sum():,}"
)

print(
    "Validation flood percentage: "
    f"{100.0 * all_targets.mean():.3f}%"
)


# ---------------------------------------------------------------------
# Select threshold using validation data only
# ---------------------------------------------------------------------

thresholds = np.round(
    np.arange(
        0.05,
        0.951,
        0.01,
    ),
    2,
)

results = []

for threshold in thresholds:
    predictions = (
        all_probabilities >= threshold
    )

    metrics = calculate_metrics(
        all_targets,
        predictions,
    )

    results.append({
        "threshold": float(threshold),
        **metrics,
    })

best_result = max(
    results,
    key=lambda result: (
        result["dice"],
        result["recall"],
        result["precision"],
    ),
)

selected_threshold = best_result["threshold"]


# ---------------------------------------------------------------------
# Save validation results
# ---------------------------------------------------------------------

with THRESHOLD_RESULTS_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys()),
    )

    writer.writeheader()
    writer.writerows(results)

threshold_summary = {
    "selected_threshold": selected_threshold,
    "selection_dataset": (
        "Rocks Road validation region"
    ),
    "selection_metric": "Dice coefficient",
    "overlap_handling": (
        "Mean probability for each unique geographic cell"
    ),
    "unique_validation_pixels": (
        unique_validation_pixels
    ),
    "duplicate_predictions_merged": (
        duplicate_predictions
    ),
    "test_dataset_used": False,
    "validation_metrics": best_result,
    "checkpoint": str(
        CHECKPOINT_PATH.relative_to(
            PROJECT_ROOT
        )
    ),
}

with SELECTED_THRESHOLD_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        threshold_summary,
        file,
        indent=4,
    )


# ---------------------------------------------------------------------
# Create threshold-selection evidence figure
# ---------------------------------------------------------------------

threshold_values = [
    result["threshold"]
    for result in results
]

dice_values = [
    result["dice"]
    for result in results
]

precision_values = [
    result["precision"]
    for result in results
]

recall_values = [
    result["recall"]
    for result in results
]

iou_values = [
    result["iou"]
    for result in results
]

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_values,
    dice_values,
    label="Dice",
    linewidth=2.5,
)

plt.plot(
    threshold_values,
    precision_values,
    label="Precision",
    linewidth=2,
)

plt.plot(
    threshold_values,
    recall_values,
    label="Recall",
    linewidth=2,
)

plt.plot(
    threshold_values,
    iou_values,
    label="IoU",
    linewidth=2,
)

plt.axvline(
    selected_threshold,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Selected threshold = "
        f"{selected_threshold:.2f}"
    ),
)

plt.scatter(
    [selected_threshold],
    [best_result["dice"]],
    color="red",
    s=70,
    zorder=5,
)

plt.title(
    "ClimateAdapt AI – Validation Threshold Selection",
    fontsize=15,
    fontweight="bold",
)

plt.xlabel("Probability threshold")
plt.ylabel("Metric score")
plt.xlim(0.05, 0.95)
plt.ylim(0.0, 1.0)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("VALIDATION THRESHOLD SELECTED")
print("=" * 72)

print(
    f"Selected threshold:  "
    f"{selected_threshold:.2f}"
)

print(
    f"Validation Dice:     "
    f"{best_result['dice']:.4f}"
)

print(
    f"Validation IoU:      "
    f"{best_result['iou']:.4f}"
)

print(
    f"Validation precision:"
    f"{best_result['precision']:.4f}"
)

print(
    f"Validation recall:   "
    f"{best_result['recall']:.4f}"
)

print(
    f"Validation accuracy: "
    f"{best_result['accuracy']:.4f}"
)

print("-" * 72)

print(
    "Results: "
    f"{THRESHOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)}"
)

print(
    "Selected threshold: "
    f"{SELECTED_THRESHOLD_PATH.relative_to(PROJECT_ROOT)}"
)

print(
    "Evidence figure: "
    f"{FIGURE_PATH.relative_to(PROJECT_ROOT)}"
)

print("=" * 72)

print(
    "The independent Tahunanui test dataset "
    "was not accessed."
)