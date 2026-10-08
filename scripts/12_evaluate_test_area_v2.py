"""Independent Experiment 2 evaluation on the held-out Tahunanui test area.

This script uses thresholds frozen on the Rocks Road validation area. It does
not search for, optimise, or modify thresholds using Tahunanui test labels.
The Waimea inlet / Monaco demonstration split is not accessed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.colors import ListedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.models.unet_v2 import CompactUNetV2  # noqa: E402


TILES_DIR = PROJECT_ROOT / "data" / "processed" / "tiles"
MANIFEST_PATH = TILES_DIR / "tile_manifest.csv"
REFERENCE_RASTER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_rasters"
    / "elevation_5m.tif"
)
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "climateadapt_unet_v2_best.pt"
THRESHOLD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation_v2"
    / "selected_threshold_v2.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_v2"
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures"

METRICS_JSON = OUTPUT_DIR / "tahunanui_test_metrics_v2.json"
METRICS_CSV = OUTPUT_DIR / "tahunanui_test_metrics_v2.csv"
PROBABILITY_RASTER = OUTPUT_DIR / "tahunanui_probability_v2.tif"
OPERATIONAL_BINARY_RASTER = (
    OUTPUT_DIR / "tahunanui_binary_operational_v2.tif"
)
DICE_BINARY_RASTER = (
    OUTPUT_DIR / "tahunanui_binary_dice_comparison_v2.tif"
)
TARGET_RASTER = OUTPUT_DIR / "tahunanui_official_flood_target_v2.tif"
ERROR_RASTER = OUTPUT_DIR / "tahunanui_prediction_errors_operational_v2.tif"
MAP_FIGURE = FIGURE_DIR / "tahunanui_test_evaluation_v2.png"
CURVES_FIGURE = FIGURE_DIR / "tahunanui_test_evaluation_curves_v2.png"


def first_present(mapping: dict, names: list[str], default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def load_thresholds(path: Path) -> tuple[float, float, dict]:
    with path.open("r", encoding="utf-8") as file:
        information = json.load(file)

    operational = first_present(
        information,
        ["operational_threshold", "selected_threshold", "threshold"],
    )
    dice_optimal = first_present(
        information,
        ["dice_optimal_threshold", "comparison_threshold"],
        default=operational,
    )

    if operational is None:
        raise KeyError(
            "The V2 validation JSON does not contain an operational threshold."
        )

    operational = float(operational)
    dice_optimal = float(dice_optimal)
    for label, value in (
        ("operational", operational),
        ("Dice-optimal", dice_optimal),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid {label} threshold: {value}")

    return operational, dice_optimal, information


def resolve_tile_path(relative_path: str) -> Path:
    normalised = str(relative_path).replace("\\", "/")
    candidates = [
        TILES_DIR / normalised,
        PROJECT_ROOT / normalised,
        Path(normalised),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate tile file: {relative_path}")


def load_tile(path: Path, expected_channels: int) -> tuple[np.ndarray, ...]:
    with np.load(path) as tile:
        feature_key = "features" if "features" in tile.files else "image"
        image = np.asarray(tile[feature_key], dtype=np.float32)
        target = np.asarray(tile["target"], dtype=np.uint8)
        valid_mask = np.asarray(tile["valid_mask"], dtype=np.uint8)

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image array in {path}; got {image.shape}")

    # Accept either channels-first or channels-last tiles.
    if image.shape[0] == expected_channels:
        channels_first = image
    elif image.shape[-1] == expected_channels:
        channels_first = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(
            f"Tile {path.name} has shape {image.shape}; expected "
            f"{expected_channels} input channels."
        )

    target = np.squeeze(target)
    valid_mask = np.squeeze(valid_mask)
    if target.shape != channels_first.shape[1:] or valid_mask.shape != target.shape:
        raise ValueError(
            f"Spatial shape mismatch in {path.name}: image={channels_first.shape}, "
            f"target={target.shape}, valid_mask={valid_mask.shape}"
        )

    return channels_first, target, valid_mask


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        cleaned[clean_key] = value
    return cleaned


def confusion_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = probability >= threshold
    positive = truth == 1
    negative = ~positive

    tp = int(np.sum(prediction & positive))
    tn = int(np.sum((~prediction) & negative))
    fp = int(np.sum(prediction & negative))
    fn = int(np.sum((~prediction) & positive))

    def safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (recall + specificity) / 2.0
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    f2 = safe_divide(5 * precision * recall, 4 * precision + recall)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "false_negative_rate": 1.0 - recall,
        "dice": dice,
        "iou": iou,
        "f2_score": f2,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def expected_calibration_error(
    truth: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (
                probability <= edges[index + 1]
            )
        else:
            selected = (probability >= edges[index]) & (
                probability < edges[index + 1]
            )
        count = int(np.sum(selected))
        if count:
            confidence = float(np.mean(probability[selected]))
            observed = float(np.mean(truth[selected]))
            error += (count / total) * abs(confidence - observed)
    return float(error)


def save_raster(
    output_path: Path,
    array: np.ndarray,
    reference_profile: dict,
    dtype: str,
    nodata,
) -> None:
    profile = reference_profile.copy()
    # Source rasters can carry strip/tile block dimensions that are not legal
    # for a newly created tiled GeoTIFF. GeoTIFF tile dimensions must be
    # multiples of 16, so replace inherited values with portable 256 px tiles.
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
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(array.astype(dtype), 1)


def evaluated_extent(mask: np.ndarray) -> tuple[slice, slice]:
    rows, columns = np.where(mask)
    if len(rows) == 0:
        raise ValueError("No evaluated test pixels were produced.")
    return (
        slice(max(0, int(rows.min()) - 10), min(mask.shape[0], int(rows.max()) + 11)),
        slice(
            max(0, int(columns.min()) - 10),
            min(mask.shape[1], int(columns.max()) + 11),
        ),
    )


def map_extent(transform, row_slice: slice, column_slice: slice) -> list[float]:
    left, top = transform * (column_slice.start, row_slice.start)
    right, bottom = transform * (column_slice.stop, row_slice.stop)
    return [left, right, bottom, top]


def metric_rows(
    operational: dict,
    dice_comparison: dict,
    threshold_free: dict,
) -> list[dict]:
    rows = []
    for group, metrics in (
        ("operational_safety_threshold", operational),
        ("dice_optimal_comparison_threshold", dice_comparison),
        ("threshold_free", threshold_free),
    ):
        for metric, value in metrics.items():
            rows.append({"metric_group": group, "metric": metric, "value": value})
    return rows


def main() -> None:
    required_paths = [
        MANIFEST_PATH,
        REFERENCE_RASTER,
        CHECKPOINT_PATH,
        THRESHOLD_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    if "split" not in manifest.columns or "file" not in manifest.columns:
        raise KeyError("The tile manifest must contain 'split' and 'file' columns.")

    split_names = manifest["split"].astype(str).str.lower().str.strip()
    if split_names.isin(["demonstration", "demo", "4"]).any():
        raise RuntimeError(
            "The tile manifest contains demonstration tiles. Experiment 2 requires "
            "Waimea/Monaco to remain untouched; regenerate tiles with split 4 excluded."
        )

    test_manifest = manifest.loc[split_names == "test"].copy()
    if test_manifest.empty:
        raise ValueError("No test tiles were found in the tile manifest.")

    operational_threshold, dice_threshold, threshold_information = load_thresholds(
        THRESHOLD_PATH
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint_configuration = checkpoint.get("configuration", {})
    input_channels = int(
        first_present(
            checkpoint,
            ["input_channels", "in_channels"],
            first_present(
                checkpoint_configuration,
                ["input_channels", "in_channels"],
                default=25,
            ),
        )
    )
    base_channels = int(
        first_present(
            checkpoint,
            ["base_channels"],
            first_present(checkpoint_configuration, ["base_channels"], default=16),
        )
    )
    dropout_probability = float(
        first_present(
            checkpoint,
            ["dropout_probability", "dropout"],
            first_present(
                checkpoint_configuration,
                ["dropout_probability", "dropout"],
                default=0.2,
            ),
        )
    )
    checkpoint_epoch = int(first_present(checkpoint, ["epoch"], default=-1))
    checkpoint_pr_auc = first_present(
        checkpoint,
        ["validation_pr_auc", "best_validation_pr_auc", "val_pr_auc"],
        default=None,
    )
    if checkpoint_pr_auc is None and isinstance(
        checkpoint.get("validation_metrics"), dict
    ):
        checkpoint_pr_auc = checkpoint["validation_metrics"].get("pr_auc")

    state_dict = first_present(
        checkpoint,
        ["model_state_dict", "state_dict", "model"],
        default=checkpoint,
    )
    if not isinstance(state_dict, dict):
        raise TypeError("Could not find a model state dictionary in the checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactUNetV2(
        input_channels=input_channels,
        base_channels=base_channels,
        dropout_probability=dropout_probability,
    ).to(device)
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()

    with rasterio.open(REFERENCE_RASTER) as reference:
        height = reference.height
        width = reference.width
        transform = reference.transform
        crs = reference.crs
        reference_profile = reference.profile.copy()

    probability_sum = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.uint16)
    target_canvas = np.zeros((height, width), dtype=np.uint8)
    valid_canvas = np.zeros((height, width), dtype=bool)
    target_written = np.zeros((height, width), dtype=bool)

    print("\n" + "=" * 78)
    print("CLIMATEADAPT EXPERIMENT 2 – INDEPENDENT TAHUNANUI TEST")
    print("=" * 78)
    print(f"Operational safety threshold: {operational_threshold:.2f}")
    print(f"Dice comparison threshold:   {dice_threshold:.2f}")
    print("Threshold source: Rocks Road validation region")
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    if checkpoint_epoch >= 0:
        print(f"Checkpoint epoch: {checkpoint_epoch}")
    if checkpoint_pr_auc is not None:
        print(f"Checkpoint validation PR-AUC: {float(checkpoint_pr_auc):.4f}")
    print(f"Test tiles: {len(test_manifest)}")
    print(f"Reference raster: {width} columns x {height} rows")
    print(f"CRS: {crs}")
    print(f"Device: {device}")
    print(f"Input channels: {input_channels}")
    print(f"Base channels: {base_channels}")
    print("Running independent Tahunanui inference...")

    with torch.inference_mode():
        for sequence, (_, row) in enumerate(test_manifest.iterrows(), start=1):
            tile_path = resolve_tile_path(str(row["file"]))
            image, target, valid_mask = load_tile(tile_path, input_channels)

            row_start = int(row["row"])
            column_start = int(row["column"])
            tile_height, tile_width = target.shape
            row_stop = min(row_start + tile_height, height)
            column_stop = min(column_start + tile_width, width)
            local_height = row_stop - row_start
            local_width = column_stop - column_start
            if local_height <= 0 or local_width <= 0:
                continue

            tensor = torch.from_numpy(image[None]).to(device)
            logits = model(tensor)
            probability = torch.sigmoid(logits).squeeze().cpu().numpy()

            probability = probability[:local_height, :local_width]
            target = target[:local_height, :local_width]
            valid_mask = valid_mask[:local_height, :local_width].astype(bool)

            destination = np.s_[row_start:row_stop, column_start:column_stop]
            probability_sum[destination] += probability * valid_mask
            prediction_count[destination] += valid_mask.astype(np.uint16)

            # Overlapping targets must agree. A maximum is safe for binary labels.
            target_canvas[destination] = np.maximum(
                target_canvas[destination], target.astype(np.uint8) * valid_mask
            )
            valid_canvas[destination] |= valid_mask
            target_written[destination] |= valid_mask

            if sequence % 25 == 0 or sequence == len(test_manifest):
                print(f"Processed test tile {sequence}/{len(test_manifest)}")

    evaluated = valid_canvas & target_written & (prediction_count > 0)
    unique_pixels = int(np.sum(evaluated))
    duplicate_predictions = int(
        np.sum(prediction_count[evaluated].astype(np.int64)) - unique_pixels
    )
    if unique_pixels == 0:
        raise RuntimeError("Inference produced no valid test pixels.")

    probability_canvas = np.full((height, width), np.nan, dtype=np.float32)
    probability_canvas[evaluated] = (
        probability_sum[evaluated] / prediction_count[evaluated]
    )

    truth = target_canvas[evaluated].astype(np.uint8)
    probability = probability_canvas[evaluated].astype(np.float64)
    if np.unique(truth).size < 2:
        raise ValueError("The test area must contain both flood and non-flood pixels.")

    operational_metrics = confusion_metrics(
        truth, probability, operational_threshold
    )
    dice_metrics = confusion_metrics(truth, probability, dice_threshold)

    fpr, tpr, _ = roc_curve(truth, probability)
    precision_curve, recall_curve, _ = precision_recall_curve(truth, probability)
    roc_auc = float(roc_auc_score(truth, probability))
    average_precision = float(average_precision_score(truth, probability))
    pr_auc_trapezoid = float(auc(recall_curve, precision_curve))
    brier = float(brier_score_loss(truth, probability))
    calibration_error = expected_calibration_error(truth, probability, bins=10)
    calibration_observed, calibration_predicted = calibration_curve(
        truth, probability, n_bins=10, strategy="uniform"
    )

    positive_pixels = int(np.sum(truth == 1))
    negative_pixels = int(np.sum(truth == 0))
    flood_percentage = 100.0 * positive_pixels / unique_pixels

    operational_prediction = np.zeros((height, width), dtype=np.uint8)
    dice_prediction = np.zeros((height, width), dtype=np.uint8)
    operational_prediction[evaluated] = (
        probability_canvas[evaluated] >= operational_threshold
    ).astype(np.uint8)
    dice_prediction[evaluated] = (
        probability_canvas[evaluated] >= dice_threshold
    ).astype(np.uint8)

    # Error codes: 0=not evaluated, 1=TN, 2=TP, 3=FP, 4=FN.
    error_canvas = np.zeros((height, width), dtype=np.uint8)
    prediction_vector = operational_prediction[evaluated].astype(bool)
    truth_vector = truth.astype(bool)
    errors = np.zeros(unique_pixels, dtype=np.uint8)
    errors[(~prediction_vector) & (~truth_vector)] = 1
    errors[prediction_vector & truth_vector] = 2
    errors[prediction_vector & (~truth_vector)] = 3
    errors[(~prediction_vector) & truth_vector] = 4
    error_canvas[evaluated] = errors

    probability_output = np.full((height, width), -9999.0, dtype=np.float32)
    probability_output[evaluated] = probability_canvas[evaluated]
    binary_operational_output = np.full((height, width), 255, dtype=np.uint8)
    binary_dice_output = np.full((height, width), 255, dtype=np.uint8)
    target_output = np.full((height, width), 255, dtype=np.uint8)
    binary_operational_output[evaluated] = operational_prediction[evaluated]
    binary_dice_output[evaluated] = dice_prediction[evaluated]
    target_output[evaluated] = target_canvas[evaluated]

    save_raster(
        PROBABILITY_RASTER,
        probability_output,
        reference_profile,
        "float32",
        -9999.0,
    )
    save_raster(
        OPERATIONAL_BINARY_RASTER,
        binary_operational_output,
        reference_profile,
        "uint8",
        255,
    )
    save_raster(
        DICE_BINARY_RASTER,
        binary_dice_output,
        reference_profile,
        "uint8",
        255,
    )
    save_raster(TARGET_RASTER, target_output, reference_profile, "uint8", 255)
    save_raster(ERROR_RASTER, error_canvas, reference_profile, "uint8", 0)

    threshold_free_metrics = {
        "roc_auc": roc_auc,
        "average_precision_pr_auc": average_precision,
        "trapezoidal_pr_auc": pr_auc_trapezoid,
        "brier_score": brier,
        "expected_calibration_error_10_bins": calibration_error,
    }

    results = {
        "experiment": "Experiment 2 – GroupNorm Compact U-Net",
        "evaluation_design": "independent geographic test",
        "test_area": "Tahunanui",
        "validation_area": "Rocks Road",
        "demonstration_area": "Waimea inlet + Monaco",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_validation_pr_auc": (
            float(checkpoint_pr_auc) if checkpoint_pr_auc is not None else None
        ),
        "input_channels": input_channels,
        "base_channels": base_channels,
        "device": str(device),
        "test_tiles": int(len(test_manifest)),
        "unique_test_pixels": unique_pixels,
        "overlapping_duplicate_predictions_merged": duplicate_predictions,
        "official_flood_pixels": positive_pixels,
        "official_non_flood_pixels": negative_pixels,
        "official_test_flood_percentage": flood_percentage,
        "threshold_source": "Rocks Road validation data",
        "threshold_modified_using_test_data": False,
        "demonstration_dataset_used": False,
        "operational_threshold_selection_rule": threshold_information.get(
            "selection_metric",
            "validation F2 score subject to the predeclared specificity rule",
        ),
        "operational_safety_threshold_metrics": operational_metrics,
        "secondary_threshold_selection_rule": "maximum validation Dice score",
        "dice_optimal_comparison_threshold_metrics": dice_metrics,
        "threshold_free_metrics": threshold_free_metrics,
        "validation_threshold_record": threshold_information,
    }

    with METRICS_JSON.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    pd.DataFrame(
        metric_rows(operational_metrics, dice_metrics, threshold_free_metrics)
    ).to_csv(METRICS_CSV, index=False)

    row_slice, column_slice = evaluated_extent(evaluated)
    extent = map_extent(transform, row_slice, column_slice)
    probability_plot = np.ma.masked_invalid(
        probability_canvas[row_slice, column_slice]
    )
    target_plot = np.ma.masked_where(
        ~evaluated[row_slice, column_slice],
        target_canvas[row_slice, column_slice],
    )
    prediction_plot = np.ma.masked_where(
        ~evaluated[row_slice, column_slice],
        operational_prediction[row_slice, column_slice],
    )
    error_plot = np.ma.masked_where(
        ~evaluated[row_slice, column_slice],
        error_canvas[row_slice, column_slice],
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    figure.suptitle(
        "ClimateAdapt AI – Experiment 2 Independent Tahunanui Test",
        fontsize=18,
        fontweight="bold",
    )

    target_image = axes[0, 0].imshow(
        target_plot,
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#f7f7f7", "#8b0000"]),
        vmin=0,
        vmax=1,
    )
    axes[0, 0].set_title("Official flood target")

    probability_image = axes[0, 1].imshow(
        probability_plot,
        extent=extent,
        origin="upper",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title("Predicted flood probability")
    figure.colorbar(probability_image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(
        prediction_plot,
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#f7f7f7", "#d62728"]),
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(
        f"Operational prediction (frozen threshold = {operational_threshold:.2f})"
    )

    error_cmap = ListedColormap(
        ["#ffffff", "#d9d9d9", "#2ca02c", "#ff7f0e", "#d62728"]
    )
    axes[1, 1].imshow(
        error_plot,
        extent=extent,
        origin="upper",
        cmap=error_cmap,
        vmin=0,
        vmax=4,
    )
    axes[1, 1].set_title("Operational errors: green=TP, orange=FP, red=FN")

    for axis in axes.flat:
        axis.set_xlabel("Easting (NZTM2000)")
        axis.set_ylabel("Northing (NZTM2000)")
        axis.ticklabel_format(style="plain", useOffset=False)

    summary = (
        f"Operational: Dice={operational_metrics['dice']:.3f}, "
        f"precision={operational_metrics['precision']:.3f}, "
        f"recall={operational_metrics['recall_sensitivity']:.3f} | "
        f"ROC-AUC={roc_auc:.3f}, AP={average_precision:.3f}"
    )
    figure.text(0.5, 0.005, summary, ha="center", fontsize=11)
    figure.savefig(MAP_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    figure.suptitle(
        "ClimateAdapt AI – Experiment 2 Tahunanui Evaluation Curves",
        fontsize=17,
        fontweight="bold",
    )

    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        recall_curve,
        precision_curve,
        color="#d62728",
        linewidth=2,
        label=f"Average precision={average_precision:.3f}",
    )
    axes[1].axhline(
        positive_pixels / unique_pixels,
        linestyle="--",
        color="grey",
        linewidth=1,
        label="Prevalence baseline",
    )
    axes[1].set_title("Precision–recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect")
    axes[2].plot(
        calibration_predicted,
        calibration_observed,
        marker="o",
        color="#2ca02c",
        linewidth=2,
        label=f"ECE={calibration_error:.3f}",
    )
    axes[2].set_title("Calibration curve")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed flood frequency")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)

    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    figure.savefig(CURVES_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Unique test pixels: {unique_pixels:,}")
    print(f"Overlapping duplicate predictions merged: {duplicate_predictions:,}")
    print(f"Official flood pixels: {positive_pixels:,}")
    print(f"Official test flood percentage: {flood_percentage:.3f}%")
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 INDEPENDENT TAHUNANUI TEST COMPLETED")
    print("=" * 78)
    print("PRIMARY – operational safety threshold frozen on Rocks Road")
    print(f"Threshold:              {operational_threshold:.2f}")
    print(f"Accuracy:               {operational_metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy:      {operational_metrics['balanced_accuracy']:.4f}"
    )
    print(f"Precision:              {operational_metrics['precision']:.4f}")
    print(f"Recall/Sensitivity:     {operational_metrics['recall_sensitivity']:.4f}")
    print(f"Specificity:            {operational_metrics['specificity']:.4f}")
    print(f"False-negative rate:    {operational_metrics['false_negative_rate']:.4f}")
    print(f"F2 score:               {operational_metrics['f2_score']:.4f}")
    print(f"Dice:                   {operational_metrics['dice']:.4f}")
    print(f"IoU:                    {operational_metrics['iou']:.4f}")
    print("-" * 78)
    print("SECONDARY – predeclared Dice-optimal comparison threshold")
    print(f"Threshold:              {dice_threshold:.2f}")
    print(f"Precision:              {dice_metrics['precision']:.4f}")
    print(f"Recall/Sensitivity:     {dice_metrics['recall_sensitivity']:.4f}")
    print(f"Dice:                   {dice_metrics['dice']:.4f}")
    print(f"IoU:                    {dice_metrics['iou']:.4f}")
    print("-" * 78)
    print(f"ROC-AUC:                {roc_auc:.4f}")
    print(f"Average precision:      {average_precision:.4f}")
    print(f"Brier score:            {brier:.4f}")
    print(f"Calibration error:      {calibration_error:.4f}")
    print("-" * 78)
    print(f"Metrics JSON: {METRICS_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Metrics CSV: {METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Probability raster: {PROBABILITY_RASTER.relative_to(PROJECT_ROOT)}")
    print(
        "Operational binary raster: "
        f"{OPERATIONAL_BINARY_RASTER.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Dice comparison raster: "
        f"{DICE_BINARY_RASTER.relative_to(PROJECT_ROOT)}"
    )
    print(f"Error raster: {ERROR_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Evaluation map: {MAP_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Evaluation curves: {CURVES_FIGURE.relative_to(PROJECT_ROOT)}")
    print("=" * 78)
    print("No threshold was selected or changed using Tahunanui test results.")
    print("Waimea inlet / Monaco demonstration data were not accessed.")


if __name__ == "__main__":
    main()
