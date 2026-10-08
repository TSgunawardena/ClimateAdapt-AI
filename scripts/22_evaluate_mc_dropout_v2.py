"""Monte Carlo dropout uncertainty evaluation for ClimateAdapt U-Net V2.

The script keeps the trained weights and Rocks Road operational threshold
frozen. It performs repeated stochastic forward passes over the independent
Tahunanui tiles, merges overlaps within each pass, and then calculates the
per-pixel predictive mean, epistemic standard deviation, and predictive
entropy. Tahunanui labels are used only for evaluation; no model parameter or
threshold is selected with test data.
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
import torch.nn as nn
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "uncertainty_v2"
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures"

MEAN_PROBABILITY_RASTER = OUTPUT_DIR / "tahunanui_mc_mean_probability_v2.tif"
EPISTEMIC_STD_RASTER = OUTPUT_DIR / "tahunanui_mc_epistemic_std_v2.tif"
PREDICTIVE_ENTROPY_RASTER = OUTPUT_DIR / "tahunanui_mc_predictive_entropy_v2.tif"
BINARY_RASTER = OUTPUT_DIR / "tahunanui_mc_binary_v2.tif"
SUMMARY_JSON = OUTPUT_DIR / "tahunanui_mc_dropout_summary_v2.json"
SUMMARY_CSV = OUTPUT_DIR / "tahunanui_mc_dropout_summary_v2.csv"
FIGURE_PATH = FIGURE_DIR / "tahunanui_mc_dropout_uncertainty_v2.png"

MC_PASSES = 20
RANDOM_SEED = 42


def first_present(mapping: dict, names: list[str], default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def load_operational_threshold(path: Path) -> tuple[float, dict]:
    with path.open("r", encoding="utf-8") as file:
        information = json.load(file)

    threshold = first_present(
        information,
        ["operational_threshold", "selected_threshold", "threshold"],
    )
    if threshold is None:
        raise KeyError("No operational threshold was found in the validation JSON.")

    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid operational threshold: {threshold}")

    return threshold, information


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
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def enable_mc_dropout(model: nn.Module) -> int:
    """Keep the model in evaluation mode but activate only dropout layers."""
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()
            count += 1
    return count


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
        "f2_score": f2,
        "dice": dice,
        "iou": iou,
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


def predictive_entropy(probability: np.ndarray) -> np.ndarray:
    """Return binary entropy normalised to the range 0..1."""
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return -(
        clipped * np.log2(clipped)
        + (1.0 - clipped) * np.log2(1.0 - clipped)
    )


def save_raster(
    output_path: Path,
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


def group_uncertainty(
    name: str,
    selected: np.ndarray,
    standard_deviation: np.ndarray,
    entropy: np.ndarray,
) -> dict[str, float | int | str]:
    count = int(np.sum(selected))
    if count == 0:
        return {
            "group": name,
            "pixels": 0,
            "mean_epistemic_std": 0.0,
            "median_epistemic_std": 0.0,
            "mean_predictive_entropy": 0.0,
        }
    return {
        "group": name,
        "pixels": count,
        "mean_epistemic_std": float(np.mean(standard_deviation[selected])),
        "median_epistemic_std": float(np.median(standard_deviation[selected])),
        "mean_predictive_entropy": float(np.mean(entropy[selected])),
    }


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
    start_time = time.perf_counter()

    manifest = pd.read_csv(MANIFEST_PATH)
    required_columns = {"split", "file", "row", "column"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise KeyError(
            "Tile manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    split_names = manifest["split"].astype(str).str.lower().str.strip()
    if split_names.isin(["demonstration", "demo", "4"]).any():
        raise RuntimeError(
            "The manifest contains demonstration tiles; regenerate it with split 4 "
            "excluded before running Experiment 2 evaluation."
        )
    test_manifest = manifest.loc[split_names == "test"].copy()
    if test_manifest.empty:
        raise ValueError("No test tiles were found in the tile manifest.")

    operational_threshold, threshold_information = load_operational_threshold(
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
    state_dict = first_present(
        checkpoint,
        ["model_state_dict", "state_dict", "model"],
        default=checkpoint,
    )
    if not isinstance(state_dict, dict):
        raise TypeError("Could not find a model state dictionary in the checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = device.type == "cuda"
    model = CompactUNetV2(
        input_channels=input_channels,
        base_channels=base_channels,
        dropout_probability=dropout_probability,
    ).to(device)
    model.load_state_dict(clean_state_dict(state_dict))
    dropout_layers = enable_mc_dropout(model)
    if dropout_layers == 0:
        raise RuntimeError(
            "No dropout layers were found. Monte Carlo dropout cannot be performed."
        )

    with rasterio.open(REFERENCE_RASTER) as reference:
        height = reference.height
        width = reference.width
        transform = reference.transform
        crs = reference.crs
        reference_profile = reference.profile.copy()

    # Load the 23 Tahunanui tiles once. This avoids repeated disk access during
    # the 20 stochastic passes and remains small enough for normal system RAM.
    tile_records = []
    target_canvas = np.zeros((height, width), dtype=np.uint8)
    valid_canvas = np.zeros((height, width), dtype=bool)

    for _, row in test_manifest.iterrows():
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

        target = target[:local_height, :local_width]
        valid_mask = valid_mask[:local_height, :local_width].astype(bool)
        destination = np.s_[row_start:row_stop, column_start:column_stop]

        target_canvas[destination] = np.maximum(
            target_canvas[destination],
            target.astype(np.uint8) * valid_mask,
        )
        valid_canvas[destination] |= valid_mask

        tile_records.append(
            {
                "image": image,
                "valid_mask": valid_mask,
                "destination": destination,
                "local_height": local_height,
                "local_width": local_width,
            }
        )

    if not tile_records:
        raise RuntimeError("No usable Tahunanui test tiles were loaded.")

    mean_canvas = np.zeros((height, width), dtype=np.float32)
    m2_canvas = np.zeros((height, width), dtype=np.float32)
    evaluated = valid_canvas.copy()

    print("\n" + "=" * 78)
    print("CLIMATEADAPT EXPERIMENT 2 – MONTE CARLO DROPOUT UNCERTAINTY")
    print("=" * 78)
    print("Area: Tahunanui independent test region")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Frozen operational threshold: {operational_threshold:.2f}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Input channels: {input_channels}")
    print(f"Dropout probability: {dropout_probability:.2f}")
    print(f"Active dropout layers: {dropout_layers}")
    print(f"Monte Carlo passes: {MC_PASSES}")
    print(f"Test tiles per pass: {len(tile_records)}")
    print("Threshold and model weights remain frozen.")

    for pass_number in range(1, MC_PASSES + 1):
        seed = RANDOM_SEED + pass_number
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        pass_sum = np.zeros((height, width), dtype=np.float32)
        pass_count = np.zeros((height, width), dtype=np.uint16)

        with torch.inference_mode():
            for record in tile_records:
                tensor = torch.from_numpy(record["image"][None]).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=mixed_precision,
                ):
                    logits = model(tensor)
                    probability = torch.sigmoid(logits)

                probability = probability.squeeze().float().cpu().numpy()
                probability = probability[
                    : record["local_height"],
                    : record["local_width"],
                ]
                valid_mask = record["valid_mask"]
                destination = record["destination"]
                pass_sum[destination] += probability * valid_mask
                pass_count[destination] += valid_mask.astype(np.uint16)

        pass_evaluated = evaluated & (pass_count > 0)
        if not np.array_equal(pass_evaluated, evaluated):
            raise RuntimeError(
                f"Pass {pass_number} did not cover every expected test pixel."
            )

        pass_probability = np.zeros((height, width), dtype=np.float32)
        pass_probability[evaluated] = (
            pass_sum[evaluated] / pass_count[evaluated]
        )

        # Welford's online algorithm avoids storing all 20 regional rasters.
        delta = pass_probability[evaluated] - mean_canvas[evaluated]
        mean_canvas[evaluated] += delta / pass_number
        delta_2 = pass_probability[evaluated] - mean_canvas[evaluated]
        m2_canvas[evaluated] += delta * delta_2

        print(f"Completed stochastic pass {pass_number:02d}/{MC_PASSES}")

    variance_canvas = np.zeros((height, width), dtype=np.float32)
    variance_canvas[evaluated] = m2_canvas[evaluated] / (MC_PASSES - 1)
    std_canvas = np.sqrt(np.maximum(variance_canvas, 0.0)).astype(np.float32)

    entropy_canvas = np.zeros((height, width), dtype=np.float32)
    entropy_canvas[evaluated] = predictive_entropy(mean_canvas[evaluated]).astype(
        np.float32
    )

    truth = target_canvas[evaluated].astype(np.uint8)
    mean_probability = mean_canvas[evaluated].astype(np.float64)
    epistemic_std = std_canvas[evaluated].astype(np.float64)
    entropy = entropy_canvas[evaluated].astype(np.float64)
    prediction = mean_probability >= operational_threshold
    correct = prediction == truth.astype(bool)
    error = ~correct

    metrics = confusion_metrics(truth, mean_probability, operational_threshold)
    metrics.update(
        {
            "roc_auc": float(roc_auc_score(truth, mean_probability)),
            "average_precision_pr_auc": float(
                average_precision_score(truth, mean_probability)
            ),
            "brier_score": float(brier_score_loss(truth, mean_probability)),
            "expected_calibration_error_10_bins": expected_calibration_error(
                truth, mean_probability, bins=10
            ),
        }
    )

    uncertainty_groups = [
        group_uncertainty("all_evaluated", np.ones_like(correct), epistemic_std, entropy),
        group_uncertainty("correct", correct, epistemic_std, entropy),
        group_uncertainty("all_errors", error, epistemic_std, entropy),
        group_uncertainty(
            "true_positive", prediction & (truth == 1), epistemic_std, entropy
        ),
        group_uncertainty(
            "true_negative", (~prediction) & (truth == 0), epistemic_std, entropy
        ),
        group_uncertainty(
            "false_positive", prediction & (truth == 0), epistemic_std, entropy
        ),
        group_uncertainty(
            "false_negative", (~prediction) & (truth == 1), epistemic_std, entropy
        ),
    ]

    high_uncertainty_cutoff = float(np.quantile(epistemic_std, 0.90))
    high_uncertainty = epistemic_std >= high_uncertainty_cutoff
    error_rate_all = float(np.mean(error))
    error_rate_high_uncertainty = float(np.mean(error[high_uncertainty]))
    error_enrichment = (
        error_rate_high_uncertainty / error_rate_all if error_rate_all else 0.0
    )
    if np.unique(error.astype(np.uint8)).size == 2:
        uncertainty_error_auc = float(
            roc_auc_score(error.astype(np.uint8), epistemic_std)
        )
    else:
        uncertainty_error_auc = 0.5

    float_nodata = -9999.0
    mean_output = np.full((height, width), float_nodata, dtype=np.float32)
    std_output = np.full((height, width), float_nodata, dtype=np.float32)
    entropy_output = np.full((height, width), float_nodata, dtype=np.float32)
    binary_output = np.full((height, width), 255, dtype=np.uint8)
    mean_output[evaluated] = mean_canvas[evaluated]
    std_output[evaluated] = std_canvas[evaluated]
    entropy_output[evaluated] = entropy_canvas[evaluated]
    binary_output[evaluated] = prediction.astype(np.uint8)

    save_raster(
        MEAN_PROBABILITY_RASTER,
        mean_output,
        reference_profile,
        "float32",
        float_nodata,
    )
    save_raster(
        EPISTEMIC_STD_RASTER,
        std_output,
        reference_profile,
        "float32",
        float_nodata,
    )
    save_raster(
        PREDICTIVE_ENTROPY_RASTER,
        entropy_output,
        reference_profile,
        "float32",
        float_nodata,
    )
    save_raster(BINARY_RASTER, binary_output, reference_profile, "uint8", 255)

    processing_seconds = time.perf_counter() - start_time
    summary = {
        "experiment": "Experiment 2 – GroupNorm Compact U-Net MC dropout",
        "method": "Monte Carlo Dropout",
        "test_area": "Tahunanui",
        "validation_area": "Rocks Road",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint_epoch,
        "input_channels": input_channels,
        "base_channels": base_channels,
        "dropout_probability": dropout_probability,
        "active_dropout_layers": dropout_layers,
        "mc_passes": MC_PASSES,
        "random_seed_start": RANDOM_SEED + 1,
        "device": str(device),
        "mixed_precision": mixed_precision,
        "unique_test_pixels": int(np.sum(evaluated)),
        "frozen_operational_threshold": operational_threshold,
        "threshold_source": "Rocks Road validation data",
        "threshold_modified_using_test_data": False,
        "model_weights_modified": False,
        "demonstration_dataset_used": False,
        "mean_probability": float(np.mean(mean_probability)),
        "mean_epistemic_standard_deviation": float(np.mean(epistemic_std)),
        "median_epistemic_standard_deviation": float(np.median(epistemic_std)),
        "maximum_epistemic_standard_deviation": float(np.max(epistemic_std)),
        "mean_predictive_entropy": float(np.mean(entropy)),
        "high_uncertainty_definition": "top 10 percent by epistemic standard deviation",
        "high_uncertainty_cutoff": high_uncertainty_cutoff,
        "overall_error_rate": error_rate_all,
        "high_uncertainty_error_rate": error_rate_high_uncertainty,
        "high_uncertainty_error_enrichment": error_enrichment,
        "uncertainty_as_error_detector_roc_auc": uncertainty_error_auc,
        "operational_metrics_using_mc_mean": metrics,
        "uncertainty_by_outcome": uncertainty_groups,
        "threshold_record": threshold_information,
        "processing_seconds": processing_seconds,
        "interpretation": (
            "Epistemic standard deviation measures variation caused by active "
            "dropout masks. Predictive entropy measures ambiguity in the mean "
            "probability. Neither measure proves that the prediction is correct."
        ),
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    pd.DataFrame(uncertainty_groups).to_csv(SUMMARY_CSV, index=False)

    row_slice, column_slice = evaluated_extent(evaluated)
    extent = map_extent(transform, row_slice, column_slice)
    evaluated_crop = evaluated[row_slice, column_slice]
    mean_plot = np.ma.masked_where(
        ~evaluated_crop, mean_canvas[row_slice, column_slice]
    )
    std_plot = np.ma.masked_where(
        ~evaluated_crop, std_canvas[row_slice, column_slice]
    )
    entropy_plot = np.ma.masked_where(
        ~evaluated_crop, entropy_canvas[row_slice, column_slice]
    )
    error_canvas = np.zeros((height, width), dtype=np.uint8)
    error_canvas[evaluated] = error.astype(np.uint8)
    error_plot = np.ma.masked_where(
        ~evaluated_crop, error_canvas[row_slice, column_slice]
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    figure.suptitle(
        "ClimateAdapt AI – Tahunanui Monte Carlo Dropout Uncertainty",
        fontsize=18,
        fontweight="bold",
    )

    mean_image = axes[0, 0].imshow(
        mean_plot, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1
    )
    axes[0, 0].set_title("Mean susceptibility probability (20 passes)")
    figure.colorbar(mean_image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    std_image = axes[0, 1].imshow(
        std_plot, extent=extent, origin="upper", cmap="magma", vmin=0
    )
    axes[0, 1].set_title("Epistemic uncertainty (dropout standard deviation)")
    figure.colorbar(std_image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    entropy_image = axes[1, 0].imshow(
        entropy_plot, extent=extent, origin="upper", cmap="plasma", vmin=0, vmax=1
    )
    axes[1, 0].set_title("Predictive entropy of mean probability")
    figure.colorbar(entropy_image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(
        error_plot,
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#d9d9d9", "#d62728"]),
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title("Prediction errors at frozen threshold (red)")

    for axis in axes.flat:
        axis.set_xlabel("Easting (NZTM2000)")
        axis.set_ylabel("Northing (NZTM2000)")
        axis.ticklabel_format(style="plain", useOffset=False)

    figure.text(
        0.5,
        0.005,
        (
            f"MC passes={MC_PASSES} | Dice={metrics['dice']:.3f} | "
            f"Recall={metrics['recall_sensitivity']:.3f} | "
            f"Uncertainty/error AUC={uncertainty_error_auc:.3f}"
        ),
        ha="center",
        fontsize=11,
    )
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("\n" + "=" * 78)
    print("MONTE CARLO DROPOUT EVALUATION COMPLETED")
    print("=" * 78)
    print(f"Unique test pixels:               {int(np.sum(evaluated)):,}")
    print(f"Mean probability:                 {np.mean(mean_probability):.4f}")
    print(f"Mean epistemic standard deviation:{np.mean(epistemic_std):.4f}")
    print(f"Mean predictive entropy:          {np.mean(entropy):.4f}")
    print(f"Accuracy:                         {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:                {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:                        {metrics['precision']:.4f}")
    print(f"Recall/Sensitivity:               {metrics['recall_sensitivity']:.4f}")
    print(f"F2 score:                         {metrics['f2_score']:.4f}")
    print(f"Dice:                             {metrics['dice']:.4f}")
    print(f"IoU:                              {metrics['iou']:.4f}")
    print(f"ROC-AUC:                          {metrics['roc_auc']:.4f}")
    print(f"Average precision:                {metrics['average_precision_pr_auc']:.4f}")
    print(f"Brier score:                      {metrics['brier_score']:.4f}")
    print("-" * 78)
    print(f"Overall error rate:               {error_rate_all:.4f}")
    print(f"Top-10% uncertainty error rate:   {error_rate_high_uncertainty:.4f}")
    print(f"Error enrichment:                 {error_enrichment:.3f}x")
    print(f"Uncertainty/error ROC-AUC:        {uncertainty_error_auc:.4f}")
    print("-" * 78)
    print(f"Summary JSON: {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Summary CSV:  {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Mean raster:  {MEAN_PROBABILITY_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Std raster:   {EPISTEMIC_STD_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Entropy:      {PREDICTIVE_ENTROPY_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Binary:       {BINARY_RASTER.relative_to(PROJECT_ROOT)}")
    print(f"Figure:       {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Processing time: {processing_seconds:.2f} seconds")
    print("=" * 78)
    print("No model weights or thresholds were changed.")
    print("Waimea inlet / Monaco demonstration data were not accessed.")


if __name__ == "__main__":
    main()
