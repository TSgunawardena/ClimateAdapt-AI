"""
Independent evaluation of the ClimateAdapt compact U-Net.

The script:
1. Loads the best checkpoint.
2. Loads the threshold selected using Rocks Road validation data.
3. Predicts the independent Tahunanui test tiles.
4. Averages overlapping tile predictions.
5. Calculates final test metrics.
6. Saves georeferenced probability, binary and error rasters.
7. Creates assessment evidence figures.

The test threshold is never adjusted in this script.
"""

from pathlib import Path
import csv
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.windows import Window, transform as window_transform
import torch

try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
except ImportError as error:
    raise ImportError(
        "Scikit-learn is required. Install it with:\n"
        "python -m pip install scikit-learn"
    ) from error


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

COMBINED_RASTER_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_rasters"
)

MANIFEST_PATH = TILES_DIR / "tile_manifest.csv"

REFERENCE_RASTER_PATH = (
    COMBINED_RASTER_DIR
    / "valid_land_mask_5m.tif"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "climateadapt_unet_best.pt"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "selected_threshold.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "test"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "evidence"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON_PATH = (
    OUTPUT_DIR
    / "tahunanui_test_metrics.json"
)

METRICS_CSV_PATH = (
    OUTPUT_DIR
    / "tahunanui_test_metrics.csv"
)

PROBABILITY_RASTER_PATH = (
    OUTPUT_DIR
    / "tahunanui_probability.tif"
)

BINARY_RASTER_PATH = (
    OUTPUT_DIR
    / "tahunanui_binary_susceptibility.tif"
)

TARGET_RASTER_PATH = (
    OUTPUT_DIR
    / "tahunanui_official_flood_target.tif"
)

ERROR_RASTER_PATH = (
    OUTPUT_DIR
    / "tahunanui_prediction_errors.tif"
)

MAP_FIGURE_PATH = (
    FIGURE_DIR
    / "tahunanui_test_evaluation.png"
)

CURVES_FIGURE_PATH = (
    FIGURE_DIR
    / "tahunanui_test_evaluation_curves.png"
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
    """Resolve a tile file recorded in the manifest."""

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

    raise FileNotFoundError(
        "Could not locate the tile file for manifest row:\n"
        f"{row}"
    )


def find_npz_array(npz_file, possible_keys, required=True):
    """Find an array inside an NPZ tile."""

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
    """Load model inputs, flood target and valid mask."""

    with np.load(tile_path) as tile:
        inputs = find_npz_array(
            tile,
            ["inputs", "input", "x", "features", "image"],
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
            ["valid_mask", "valid", "land_mask"],
            required=False,
        )

    inputs = np.asarray(inputs, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).squeeze()

    if inputs.ndim != 3:
        raise ValueError(
            f"Expected three-dimensional inputs in {tile_path}; "
            f"received {inputs.shape}"
        )

    if inputs.shape[0] != 24 and inputs.shape[-1] == 24:
        inputs = np.transpose(inputs, (2, 0, 1))

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

    return inputs, target, valid_mask


def extract_state_dict(checkpoint):
    """Extract model weights from different checkpoint formats."""

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
    """Remove DataParallel prefixes."""

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    return cleaned


def infer_model_dimensions(state_dict):
    """Infer input and base-channel counts."""

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

    return (
        int(first_weight.shape[1]),
        int(first_weight.shape[0]),
    )


def calculate_binary_metrics(target, prediction):
    """Calculate threshold-dependent segmentation metrics."""

    target = target.astype(bool)
    prediction = prediction.astype(bool)

    true_positive = int(
        np.logical_and(target, prediction).sum()
    )

    true_negative = int(
        np.logical_and(~target, ~prediction).sum()
    )

    false_positive = int(
        np.logical_and(~target, prediction).sum()
    )

    false_negative = int(
        np.logical_and(target, ~prediction).sum()
    )

    epsilon = 1e-8

    precision = true_positive / (
        true_positive + false_positive + epsilon
    )

    recall = true_positive / (
        true_positive + false_negative + epsilon
    )

    specificity = true_negative / (
        true_negative + false_positive + epsilon
    )

    false_negative_rate = false_negative / (
        false_negative + true_positive + epsilon
    )

    false_positive_rate = false_positive / (
        false_positive + true_negative + epsilon
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

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "false_negative_rate": float(false_negative_rate),
        "false_positive_rate": float(false_positive_rate),
        "dice": float(dice),
        "iou": float(iou),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def calculate_expected_calibration_error(
    target,
    probabilities,
    number_of_bins=10,
):
    """Calculate expected calibration error."""

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    bin_ids = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=False,
    )

    expected_calibration_error = 0.0
    total_count = len(target)

    for bin_id in range(number_of_bins):
        members = bin_ids == bin_id

        if not np.any(members):
            continue

        bin_accuracy = target[members].mean()
        bin_confidence = probabilities[members].mean()
        bin_weight = members.sum() / total_count

        expected_calibration_error += (
            bin_weight
            * abs(bin_accuracy - bin_confidence)
        )

    return float(expected_calibration_error)


def save_raster(
    output_path,
    array,
    reference_profile,
    dtype,
    nodata,
):
    """Save a single-band georeferenced raster."""

    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as destination:
        destination.write(
            array.astype(dtype),
            1,
        )


# ---------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------

print()
print("=" * 76)
print("CLIMATEADAPT INDEPENDENT TAHUNANUI TEST EVALUATION")
print("=" * 76)

for required_path in [
    MANIFEST_PATH,
    REFERENCE_RASTER_PATH,
    CHECKPOINT_PATH,
    THRESHOLD_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {required_path}"
        )


# ---------------------------------------------------------------------
# Load frozen validation threshold
# ---------------------------------------------------------------------

with THRESHOLD_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    threshold_information = json.load(file)

selected_threshold = float(
    threshold_information["selected_threshold"]
)

print(
    f"Frozen validation threshold: "
    f"{selected_threshold:.2f}"
)

print(
    "Threshold source: Rocks Road validation region"
)


# ---------------------------------------------------------------------
# Load test manifest rows
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
    ["split", "dataset_split", "set", "subset"],
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
)

row_column = find_column(
    fieldnames,
    ["row", "row_offset", "row_off", "y"],
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

test_rows = [
    row
    for row in manifest_rows
    if row[split_column].strip().lower()
    in {"test", "testing"}
]

if not test_rows:
    raise RuntimeError(
        "No test tiles were found in the manifest."
    )

print(f"Test tiles: {len(test_rows)}")


# ---------------------------------------------------------------------
# Load raster metadata
# ---------------------------------------------------------------------

with rasterio.open(
    REFERENCE_RASTER_PATH
) as reference_raster:
    reference_profile = reference_raster.profile.copy()
    raster_height = reference_raster.height
    raster_width = reference_raster.width
    raster_transform = reference_raster.transform
    raster_crs = reference_raster.crs

print(
    f"Reference raster: "
    f"{raster_width} columns x "
    f"{raster_height} rows"
)

print(f"CRS: {raster_crs}")


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

state_dict = clean_state_dict(
    extract_state_dict(checkpoint)
)

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
# Predict test tiles and merge overlaps
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

print("Running independent Tahunanui test inference...")

with torch.inference_mode():
    for tile_number, row in enumerate(
        test_rows,
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

        probability_view[valid_pixels] += (
            probabilities[valid_pixels]
        )

        count_view[valid_pixels] += 1

        target_view[valid_pixels] = np.maximum(
            target_view[valid_pixels],
            target[valid_pixels].astype(np.uint8),
        )

        print(
            f"\rProcessed test tile "
            f"{tile_number}/{len(test_rows)}",
            end="",
        )

print()


# ---------------------------------------------------------------------
# Create unique-pixel test arrays
# ---------------------------------------------------------------------

evaluated_pixels = prediction_count > 0

if not evaluated_pixels.any():
    raise RuntimeError(
        "No valid test pixels were evaluated."
    )

average_probability = np.zeros(
    (raster_height, raster_width),
    dtype=np.float32,
)

average_probability[evaluated_pixels] = (
    probability_sum[evaluated_pixels]
    / prediction_count[evaluated_pixels]
)

binary_prediction = np.zeros(
    (raster_height, raster_width),
    dtype=np.uint8,
)

binary_prediction[evaluated_pixels] = (
    average_probability[evaluated_pixels]
    >= selected_threshold
).astype(np.uint8)

test_probabilities = average_probability[
    evaluated_pixels
].astype(np.float32)

test_targets = target_canvas[
    evaluated_pixels
].astype(np.uint8)

test_predictions = binary_prediction[
    evaluated_pixels
].astype(np.uint8)

unique_test_pixels = int(
    evaluated_pixels.sum()
)

total_tile_predictions = int(
    prediction_count[evaluated_pixels].sum()
)

duplicates_merged = (
    total_tile_predictions
    - unique_test_pixels
)

print(f"Unique test pixels: {unique_test_pixels:,}")

print(
    f"Overlapping duplicate predictions merged: "
    f"{duplicates_merged:,}"
)

print(
    f"Official flood pixels: "
    f"{test_targets.sum():,}"
)

print(
    f"Official test flood percentage: "
    f"{100.0 * test_targets.mean():.3f}%"
)


# ---------------------------------------------------------------------
# Calculate final independent test metrics
# ---------------------------------------------------------------------

metrics = calculate_binary_metrics(
    test_targets,
    test_predictions,
)

roc_auc = float(
    roc_auc_score(
        test_targets,
        test_probabilities,
    )
)

average_precision = float(
    average_precision_score(
        test_targets,
        test_probabilities,
    )
)

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        test_targets,
        test_probabilities,
    )
)

precision_recall_auc = float(
    auc(
        recall_curve,
        precision_curve,
    )
)

false_positive_rates, true_positive_rates, _ = (
    roc_curve(
        test_targets,
        test_probabilities,
    )
)

brier_score = float(
    np.mean(
        (
            test_probabilities
            - test_targets
        ) ** 2
    )
)

expected_calibration_error = (
    calculate_expected_calibration_error(
        test_targets,
        test_probabilities,
        number_of_bins=10,
    )
)

metrics.update({
    "roc_auc": roc_auc,
    "precision_recall_auc": precision_recall_auc,
    "average_precision": average_precision,
    "brier_score": brier_score,
    "expected_calibration_error": (
        expected_calibration_error
    ),
})


# ---------------------------------------------------------------------
# Create output raster arrays
# ---------------------------------------------------------------------

probability_output = np.full(
    (raster_height, raster_width),
    -9999.0,
    dtype=np.float32,
)

probability_output[evaluated_pixels] = (
    average_probability[evaluated_pixels]
)

binary_output = np.full(
    (raster_height, raster_width),
    255,
    dtype=np.uint8,
)

binary_output[evaluated_pixels] = (
    binary_prediction[evaluated_pixels]
)

target_output = np.full(
    (raster_height, raster_width),
    255,
    dtype=np.uint8,
)

target_output[evaluated_pixels] = (
    target_canvas[evaluated_pixels]
)

# Error codes:
# 0 = true negative
# 1 = true positive
# 2 = false positive
# 3 = false negative
# 255 = not evaluated

error_output = np.full(
    (raster_height, raster_width),
    255,
    dtype=np.uint8,
)

target_boolean = target_canvas.astype(bool)
prediction_boolean = binary_prediction.astype(bool)

error_output[
    evaluated_pixels
    & ~target_boolean
    & ~prediction_boolean
] = 0

error_output[
    evaluated_pixels
    & target_boolean
    & prediction_boolean
] = 1

error_output[
    evaluated_pixels
    & ~target_boolean
    & prediction_boolean
] = 2

error_output[
    evaluated_pixels
    & target_boolean
    & ~prediction_boolean
] = 3


# ---------------------------------------------------------------------
# Save georeferenced output rasters
# ---------------------------------------------------------------------

save_raster(
    PROBABILITY_RASTER_PATH,
    probability_output,
    reference_profile,
    "float32",
    -9999.0,
)

save_raster(
    BINARY_RASTER_PATH,
    binary_output,
    reference_profile,
    "uint8",
    255,
)

save_raster(
    TARGET_RASTER_PATH,
    target_output,
    reference_profile,
    "uint8",
    255,
)

save_raster(
    ERROR_RASTER_PATH,
    error_output,
    reference_profile,
    "uint8",
    255,
)


# ---------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------

result_summary = {
    "evaluation_area": "Tahunanui, Nelson",
    "evaluation_type": "Independent geographic test",
    "checkpoint": str(
        CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
    ),
    "threshold": selected_threshold,
    "threshold_source": (
        "Rocks Road validation region"
    ),
    "threshold_modified_using_test_data": False,
    "overlap_handling": (
        "Mean probability for each unique geographic cell"
    ),
    "unique_test_pixels": unique_test_pixels,
    "duplicates_merged": duplicates_merged,
    "official_flood_pixels": int(
        test_targets.sum()
    ),
    "official_flood_percentage": float(
        100.0 * test_targets.mean()
    ),
    "metrics": metrics,
}

with METRICS_JSON_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result_summary,
        file,
        indent=4,
    )

with METRICS_CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "metric",
        "value",
    ])

    writer.writerow([
        "selected_threshold",
        selected_threshold,
    ])

    writer.writerow([
        "unique_test_pixels",
        unique_test_pixels,
    ])

    writer.writerow([
        "official_flood_pixels",
        int(test_targets.sum()),
    ])

    for metric_name, metric_value in metrics.items():
        writer.writerow([
            metric_name,
            metric_value,
        ])


# ---------------------------------------------------------------------
# Create geographic evaluation figure
# ---------------------------------------------------------------------

evaluated_rows, evaluated_columns = np.where(
    evaluated_pixels
)

padding = 10

row_min = max(
    0,
    int(evaluated_rows.min()) - padding,
)

row_max = min(
    raster_height,
    int(evaluated_rows.max()) + padding + 1,
)

column_min = max(
    0,
    int(evaluated_columns.min()) - padding,
)

column_max = min(
    raster_width,
    int(evaluated_columns.max()) + padding + 1,
)

crop_height = row_max - row_min
crop_width = column_max - column_min

crop_window = Window(
    column_min,
    row_min,
    crop_width,
    crop_height,
)

crop_transform = window_transform(
    crop_window,
    raster_transform,
)

west, south, east, north = array_bounds(
    crop_height,
    crop_width,
    crop_transform,
)

extent = [
    west,
    east,
    south,
    north,
]

crop_evaluated = evaluated_pixels[
    row_min:row_max,
    column_min:column_max,
]

crop_target = np.ma.masked_where(
    ~crop_evaluated,
    target_canvas[
        row_min:row_max,
        column_min:column_max,
    ],
)

crop_probability = np.ma.masked_where(
    ~crop_evaluated,
    average_probability[
        row_min:row_max,
        column_min:column_max,
    ],
)

crop_binary = np.ma.masked_where(
    ~crop_evaluated,
    binary_prediction[
        row_min:row_max,
        column_min:column_max,
    ],
)

crop_error = np.ma.masked_where(
    ~crop_evaluated,
    error_output[
        row_min:row_max,
        column_min:column_max,
    ],
)

figure, axes = plt.subplots(
    2,
    2,
    figsize=(15, 12),
)

figure.suptitle(
    "ClimateAdapt AI – Independent Tahunanui Test Evaluation",
    fontsize=18,
    fontweight="bold",
)

axes[0, 0].imshow(
    crop_target,
    cmap="Reds",
    vmin=0,
    vmax=1,
    extent=extent,
    origin="upper",
)

axes[0, 0].set_title(
    "Official Flood Target"
)

probability_image = axes[0, 1].imshow(
    crop_probability,
    cmap="viridis",
    vmin=0,
    vmax=1,
    extent=extent,
    origin="upper",
)

axes[0, 1].set_title(
    "Predicted Flood Probability"
)

figure.colorbar(
    probability_image,
    ax=axes[0, 1],
    fraction=0.046,
    pad=0.04,
    label="Probability",
)

axes[1, 0].imshow(
    crop_binary,
    cmap="Blues",
    vmin=0,
    vmax=1,
    extent=extent,
    origin="upper",
)

axes[1, 0].set_title(
    f"Binary Susceptibility Map "
    f"(threshold = {selected_threshold:.2f})"
)

error_colormap = ListedColormap([
    "#e5e7eb",
    "#16a34a",
    "#f59e0b",
    "#dc2626",
])

axes[1, 1].imshow(
    crop_error,
    cmap=error_colormap,
    vmin=0,
    vmax=3,
    extent=extent,
    origin="upper",
)

axes[1, 1].set_title(
    "Prediction Outcomes"
)

error_legend = [
    Patch(
        facecolor="#e5e7eb",
        label="True negative",
    ),
    Patch(
        facecolor="#16a34a",
        label="True positive",
    ),
    Patch(
        facecolor="#f59e0b",
        label="False positive",
    ),
    Patch(
        facecolor="#dc2626",
        label="False negative",
    ),
]

axes[1, 1].legend(
    handles=error_legend,
    loc="lower right",
)

for axis in axes.flat:
    axis.set_xlabel("Easting (NZTM2000)")
    axis.set_ylabel("Northing (NZTM2000)")
    axis.ticklabel_format(
        style="plain",
        axis="both",
    )

metrics_text = (
    f"Dice: {metrics['dice']:.3f}   "
    f"IoU: {metrics['iou']:.3f}   "
    f"Precision: {metrics['precision']:.3f}   "
    f"Recall: {metrics['recall_sensitivity']:.3f}   "
    f"ROC-AUC: {metrics['roc_auc']:.3f}   "
    f"PR-AUC: {metrics['precision_recall_auc']:.3f}"
)

figure.text(
    0.5,
    0.02,
    metrics_text,
    ha="center",
    fontsize=11,
)

plt.tight_layout(
    rect=[0, 0.04, 1, 0.96]
)

plt.savefig(
    MAP_FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Create ROC, precision-recall and calibration curves
# ---------------------------------------------------------------------

calibration_true, calibration_predicted = (
    calibration_curve(
        test_targets,
        test_probabilities,
        n_bins=10,
        strategy="quantile",
    )
)

figure, axes = plt.subplots(
    1,
    3,
    figsize=(17, 5),
)

axes[0].plot(
    false_positive_rates,
    true_positive_rates,
    linewidth=2.5,
    label=f"ROC-AUC = {roc_auc:.3f}",
)

axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="grey",
)

axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False-positive rate")
axes[0].set_ylabel("True-positive rate")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(
    recall_curve,
    precision_curve,
    linewidth=2.5,
    label=f"PR-AUC = {precision_recall_auc:.3f}",
)

axes[1].axhline(
    test_targets.mean(),
    linestyle="--",
    color="grey",
    label="Flood prevalence",
)

axes[1].set_title("Precision–Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
axes[1].grid(alpha=0.25)

axes[2].plot(
    calibration_predicted,
    calibration_true,
    marker="o",
    linewidth=2.5,
    label="U-Net",
)

axes[2].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="grey",
    label="Perfect calibration",
)

axes[2].set_title(
    f"Calibration Curve\n"
    f"Brier = {brier_score:.3f}, "
    f"ECE = {expected_calibration_error:.3f}"
)

axes[2].set_xlabel("Mean predicted probability")
axes[2].set_ylabel("Observed flood frequency")
axes[2].legend()
axes[2].grid(alpha=0.25)

figure.suptitle(
    "ClimateAdapt AI – Tahunanui Test Performance",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()

plt.savefig(
    CURVES_FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Final terminal summary
# ---------------------------------------------------------------------

print()
print("=" * 76)
print("INDEPENDENT TAHUNANUI TEST COMPLETED")
print("=" * 76)

print(f"Frozen threshold:       {selected_threshold:.2f}")
print(f"Accuracy:               {metrics['accuracy']:.4f}")
print(f"Balanced accuracy:      {metrics['balanced_accuracy']:.4f}")
print(f"Precision:              {metrics['precision']:.4f}")
print(f"Recall/Sensitivity:     {metrics['recall_sensitivity']:.4f}")
print(f"Specificity:            {metrics['specificity']:.4f}")
print(f"False-negative rate:    {metrics['false_negative_rate']:.4f}")
print(f"Dice:                   {metrics['dice']:.4f}")
print(f"IoU:                    {metrics['iou']:.4f}")
print(f"ROC-AUC:                {metrics['roc_auc']:.4f}")
print(f"PR-AUC:                 {metrics['precision_recall_auc']:.4f}")
print(f"Average precision:      {metrics['average_precision']:.4f}")
print(f"Brier score:            {metrics['brier_score']:.4f}")
print(f"Calibration error:      {metrics['expected_calibration_error']:.4f}")

print("-" * 76)

print(
    f"True positives:         "
    f"{metrics['true_positive']:,}"
)

print(
    f"True negatives:         "
    f"{metrics['true_negative']:,}"
)

print(
    f"False positives:        "
    f"{metrics['false_positive']:,}"
)

print(
    f"False negatives:        "
    f"{metrics['false_negative']:,}"
)

print("-" * 76)

print(
    "Metrics JSON: "
    f"{METRICS_JSON_PATH.relative_to(PROJECT_ROOT)}"
)

print(
    "Metrics CSV: "
    f"{METRICS_CSV_PATH.relative_to(PROJECT_ROOT)}"
)

print(
    "Probability raster: "
    f"{PROBABILITY_RASTER_PATH.relative_to(PROJECT_ROOT)}"
)

print(
    "Binary raster: "
    f"{BINARY_RASTER_PATH.relative_to(PROJECT_ROOT)}"
)

print(
    "Error raster: "
    f"{ERROR_RASTER_PATH.relative_to(PROJECT_ROOT)}"
)

print(
    "Evaluation map: "
    f"{MAP_FIGURE_PATH.relative_to(PROJECT_ROOT)}"
)

print(
    "Evaluation curves: "
    f"{CURVES_FIGURE_PATH.relative_to(PROJECT_ROOT)}"
)

print("=" * 76)

print(
    "The threshold was frozen using validation data "
    "and was not adjusted using test results."
)