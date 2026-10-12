"""Final, one-time evaluation of the frozen ClimateAdapt V3 models.

This script opens only the locked V3 test tiles (Delaware Open Coast and
Oananga Bay), evaluates thresholds selected on validation data, and writes
pooled plus per-area evidence. It never trains a model or selects a threshold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.models.unet_v3 import CompactUNetV3  # noqa: E402


TILES_DIR = PROJECT_ROOT / "data" / "processed" / "v3" / "tiles"
MANIFEST_PATH = TILES_DIR / "v3_tile_manifest.csv"
REFERENCE_RASTER = (
    PROJECT_ROOT / "data" / "processed" / "v3" / "v3_split_mask_5m.tif"
)
UNET_CHECKPOINT = PROJECT_ROOT / "models" / "v3" / "climateadapt_unet_v3_best.pt"
UNET_THRESHOLD_PATH = (
    PROJECT_ROOT / "outputs" / "validation_v3" / "selected_threshold_v3.json"
)
LOGISTIC_MODEL_PATH = (
    PROJECT_ROOT / "models" / "v3" / "logistic_regression_v3.joblib"
)
LOGISTIC_THRESHOLD_PATH = (
    PROJECT_ROOT / "outputs" / "baselines_v3" / "logistic_selected_threshold_v3.json"
)
BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "baselines_v3" / "baseline_validation_summary_v3.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_v3"
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures" / "v3"

SUMMARY_JSON = OUTPUT_DIR / "v3_locked_test_summary.json"
POOLED_CSV = OUTPUT_DIR / "v3_locked_test_model_comparison.csv"
AREA_CSV = OUTPUT_DIR / "v3_locked_test_area_metrics.csv"
UNET_PROBABILITY_RASTER = OUTPUT_DIR / "v3_test_unet_probability.tif"
UNET_BINARY_RASTER = OUTPUT_DIR / "v3_test_unet_operational.tif"
LOGISTIC_PROBABILITY_RASTER = OUTPUT_DIR / "v3_test_logistic_probability.tif"
LOGISTIC_BINARY_RASTER = OUTPUT_DIR / "v3_test_logistic_operational.tif"
TARGET_RASTER = OUTPUT_DIR / "v3_test_official_target.tif"
UNET_ERROR_RASTER = OUTPUT_DIR / "v3_test_unet_errors.tif"
MAP_FIGURE = FIGURE_DIR / "v3_locked_test_evaluation.png"
CURVES_FIGURE = FIGURE_DIR / "v3_locked_test_curves.png"

AREA_NAMES = {3: "Delaware Open Coast", 6: "Oananga Bay"}
EXPECTED_TEST_AREAS = set(AREA_NAMES)
RANDOM_SEED = 8532


def first_present(mapping: dict, names: list[str], default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def find_nested(mapping, names: tuple[str, ...], default=None):
    """Find a named scalar in a small JSON/joblib metadata tree."""
    if isinstance(mapping, dict):
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
        for value in mapping.values():
            found = find_nested(value, names, default=None)
            if found is not None:
                return found
    return default


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        name = str(key)
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def resolve_tile_path(relative_path: str) -> Path:
    normalised = str(relative_path).replace("\\", "/")
    candidates = (
        TILES_DIR / normalised,
        PROJECT_ROOT / normalised,
        Path(normalised),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate V3 tile: {relative_path}")


def array_from_any(tile, keys: tuple[str, ...], path: Path) -> np.ndarray:
    for key in keys:
        if key in tile.files:
            return np.asarray(tile[key])
    raise KeyError(f"None of {keys} was found in {path.name}; keys={tile.files}")


def load_tile(path: Path, expected_channels: int) -> tuple[np.ndarray, ...]:
    with np.load(path) as tile:
        image = array_from_any(tile, ("features", "image", "x"), path).astype(
            np.float32
        )
        target = array_from_any(tile, ("target", "y", "mask"), path).astype(
            np.uint8
        )
        valid = array_from_any(tile, ("valid_mask", "valid"), path).astype(bool)
        if "area_zone_id" in tile.files:
            area = np.asarray(tile["area_zone_id"], dtype=np.uint8)
        elif "area_id" in tile.files:
            area = np.asarray(tile["area_id"], dtype=np.uint8)
        else:
            area = np.zeros(np.squeeze(target).shape, dtype=np.uint8)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D features in {path}; got {image.shape}")
    if image.shape[0] == expected_channels:
        channels_first = image
    elif image.shape[-1] == expected_channels:
        channels_first = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(
            f"{path.name} has {image.shape}; expected {expected_channels} channels"
        )

    target = np.squeeze(target)
    valid = np.squeeze(valid)
    area = np.squeeze(area)
    shape = channels_first.shape[1:]
    if target.shape != shape or valid.shape != shape or area.shape != shape:
        raise ValueError(
            f"Spatial mismatch in {path.name}: features={shape}, target={target.shape}, "
            f"valid={valid.shape}, area={area.shape}"
        )
    return channels_first, target, valid, area


def unwrap_estimator(artifact):
    if hasattr(artifact, "predict_proba"):
        return artifact, {}
    if isinstance(artifact, dict):
        for key in ("model", "pipeline", "estimator", "classifier"):
            candidate = artifact.get(key)
            if hasattr(candidate, "predict_proba"):
                return candidate, artifact
    raise TypeError("The logistic joblib does not contain a predict_proba estimator.")


def threshold_from_json(path: Path, names: tuple[str, ...]) -> tuple[float, dict]:
    information = read_json(path)
    value = first_present(information, list(names))
    if value is None:
        raise KeyError(f"No threshold key {names} in {path}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid threshold {value} in {path}")
    return value, information


def metric_record(
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    model: str,
    scope: str,
) -> dict:
    prediction = (probability >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    recall = recall_score(truth, prediction, zero_division=0)
    dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    has_both = np.unique(truth).size == 2
    return {
        "scope": scope,
        "model": model,
        "threshold": float(threshold),
        "pixels": int(truth.size),
        "positive_pixels": int(np.sum(truth)),
        "prevalence": float(np.mean(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "false_negative_rate": float(1.0 - recall),
        "f2_score": float(fbeta_score(truth, prediction, beta=2, zero_division=0)),
        "dice": float(dice),
        "iou": float(iou),
        "roc_auc": float(roc_auc_score(truth, probability)) if has_both else None,
        "average_precision": (
            float(average_precision_score(truth, probability)) if has_both else None
        ),
        "brier_score": float(brier_score_loss(truth, probability)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def merge_overlaps(
    indices: np.ndarray,
    truth: np.ndarray,
    areas: np.ndarray,
    unet_probability: np.ndarray,
    logistic_probability: np.ndarray,
) -> tuple[np.ndarray, ...]:
    order = np.argsort(indices, kind="mergesort")
    indices = indices[order]
    truth = truth[order]
    areas = areas[order]
    unet_probability = unet_probability[order]
    logistic_probability = logistic_probability[order]

    unique, starts, counts = np.unique(
        indices, return_index=True, return_counts=True
    )
    truth_min = np.minimum.reduceat(truth, starts)
    truth_max = np.maximum.reduceat(truth, starts)
    area_min = np.minimum.reduceat(areas, starts)
    area_max = np.maximum.reduceat(areas, starts)
    if not np.array_equal(truth_min, truth_max):
        raise RuntimeError("Overlapping test tiles disagree on target labels.")
    if not np.array_equal(area_min, area_max):
        raise RuntimeError("Overlapping test tiles disagree on area IDs.")

    unet_mean = np.add.reduceat(unet_probability, starts) / counts
    logistic_mean = np.add.reduceat(logistic_probability, starts) / counts
    duplicates = int(indices.size - unique.size)
    return (
        unique,
        truth_max.astype(np.uint8),
        area_max.astype(np.uint8),
        unet_mean.astype(np.float32),
        logistic_mean.astype(np.float32),
        duplicates,
    )


def save_indexed_raster(
    path: Path,
    indices: np.ndarray,
    values: np.ndarray,
    profile: dict,
    dtype: str,
    nodata,
) -> None:
    height = int(profile["height"])
    width = int(profile["width"])
    canvas = np.full(height * width, nodata, dtype=np.dtype(dtype))
    canvas[indices] = values.astype(dtype, copy=False)
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="DEFLATE",
        predictor=2 if dtype.startswith("float") else 1,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(canvas.reshape(height, width), 1)


def create_map_figure(
    indices: np.ndarray,
    truth: np.ndarray,
    areas: np.ndarray,
    unet_probability: np.ndarray,
    logistic_probability: np.ndarray,
    unet_threshold: float,
    logistic_threshold: float,
    width: int,
) -> None:
    rows = indices // width
    columns = indices % width
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for plot_row, area_id in enumerate(sorted(AREA_NAMES)):
        selector = areas == area_id
        selected = np.flatnonzero(selector)
        # Rasterized scatter is used so two spatially separated test regions can
        # be shown without allocating another full 75-million-cell canvas.
        # A fixed subsample keeps the evidence figure quick and reproducible.
        if selected.size > 150_000:
            rng = np.random.default_rng(RANDOM_SEED + area_id)
            selected = np.sort(rng.choice(selected, 150_000, replace=False))
        x = columns[selected]
        y = -rows[selected]
        true = truth[selected]
        unet = unet_probability[selected]
        logistic = logistic_probability[selected]
        unet_error = (unet >= unet_threshold).astype(np.uint8) - true.astype(np.int8)
        logistic_error = (
            (logistic >= logistic_threshold).astype(np.uint8) - true.astype(np.int8)
        )
        panels = (
            (true, "Official target", "Reds", 0, 1),
            (unet, "U-Net probability", "viridis", 0, 1),
            (unet_error, "U-Net error", "bwr", -1, 1),
            (logistic_error, "Logistic error", "bwr", -1, 1),
        )
        for axis, (colour, title, cmap, vmin, vmax) in zip(axes[plot_row], panels):
            image = axis.scatter(
                x, y, c=colour, s=1.0, cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True,
            )
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.set_axis_off()
            figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
        axes[plot_row, 0].text(
            0.0, 1.08, AREA_NAMES[area_id], transform=axes[plot_row, 0].transAxes,
            fontsize=13, fontweight="bold"
        )
    figure.suptitle(
        "ClimateAdapt V3 – Locked Geographic Test Evaluation", fontsize=18,
        fontweight="bold"
    )
    figure.savefig(MAP_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_curve_figure(
    truth: np.ndarray,
    unet_probability: np.ndarray,
    logistic_probability: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, probability, colour in (
        ("Frozen U-Net V3", unet_probability, "#1769aa"),
        ("Frozen logistic", logistic_probability, "#e07a1f"),
    ):
        precision, recall, _ = precision_recall_curve(truth, probability)
        ap = average_precision_score(truth, probability)
        axes[0].plot(recall, precision, lw=2, color=colour, label=f"{name} (AP={ap:.3f})")
        fpr, tpr, _ = roc_curve(truth, probability)
        auc = roc_auc_score(truth, probability)
        axes[1].plot(fpr, tpr, lw=2, color=colour, label=f"{name} (AUC={auc:.3f})")
    axes[0].axhline(np.mean(truth), color="grey", ls="--", label="Test prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    axes[1].plot([0, 1], [0, 1], color="grey", ls="--")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle("V3 frozen models on the locked test regions", fontsize=16, fontweight="bold")
    figure.savefig(CURVES_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    required = (
        MANIFEST_PATH,
        REFERENCE_RASTER,
        UNET_CHECKPOINT,
        UNET_THRESHOLD_PATH,
        LOGISTIC_MODEL_PATH,
        BASELINE_SUMMARY_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required V3 artifacts:\n  " + "\n  ".join(missing))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    if not {"split", "file", "row", "column"}.issubset(manifest.columns):
        raise KeyError("V3 manifest must contain split, file, row and column.")
    test_manifest = manifest[
        manifest["split"].astype(str).str.lower().str.strip().isin(("test", "testing", "3"))
    ].copy()
    if test_manifest.empty:
        raise RuntimeError("No locked test tiles were found.")

    unet_threshold, unet_threshold_info = threshold_from_json(
        UNET_THRESHOLD_PATH,
        ("operational_threshold", "selected_threshold", "threshold"),
    )
    dice_threshold = float(
        first_present(
            unet_threshold_info,
            ["dice_optimal_threshold", "comparison_threshold"],
            unet_threshold,
        )
    )
    baseline_summary = read_json(BASELINE_SUMMARY_PATH)
    if LOGISTIC_THRESHOLD_PATH.exists():
        logistic_threshold, logistic_threshold_info = threshold_from_json(
            LOGISTIC_THRESHOLD_PATH,
            ("selected_threshold", "operational_threshold", "threshold"),
        )
    else:
        # Script 41 records the frozen value in its main validation summary.
        value = find_nested(
            baseline_summary,
            ("selected_threshold", "logistic_threshold", "operational_threshold"),
        )
        if value is None:
            raise KeyError(
                "Could not find the frozen logistic validation threshold in "
                f"{BASELINE_SUMMARY_PATH}."
            )
        logistic_threshold = float(value)
        logistic_threshold_info = baseline_summary

    checkpoint = torch.load(UNET_CHECKPOINT, map_location="cpu", weights_only=False)
    configuration = checkpoint.get("configuration", {}) if isinstance(checkpoint, dict) else {}
    input_channels = int(
        first_present(checkpoint, ["input_channels", "in_channels"],
                      first_present(configuration, ["input_channels", "in_channels"], 11))
    )
    base_channels = int(
        first_present(checkpoint, ["base_channels"],
                      first_present(configuration, ["base_channels"], 16))
    )
    dropout = float(
        first_present(checkpoint, ["dropout_probability", "dropout"],
                      first_present(configuration, ["dropout_probability", "dropout"], 0.2))
    )
    checkpoint_epoch = int(first_present(checkpoint, ["epoch"], -1))
    checkpoint_ap = first_present(
        checkpoint, ["validation_pr_auc", "best_validation_pr_auc", "val_pr_auc"]
    )
    if checkpoint_ap is None and isinstance(checkpoint.get("validation_metrics"), dict):
        checkpoint_ap = checkpoint["validation_metrics"].get("pr_auc")
    state_dict = first_present(checkpoint, ["model_state_dict", "state_dict", "model"], checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactUNetV3(
        input_channels=input_channels,
        base_channels=base_channels,
        dropout_probability=dropout,
    ).to(device)
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()

    logistic_artifact = joblib.load(LOGISTIC_MODEL_PATH)
    logistic_model, logistic_metadata = unwrap_estimator(logistic_artifact)
    training_prevalence = first_present(
        logistic_metadata,
        ["training_prevalence", "train_prevalence"],
        find_nested(
            baseline_summary,
            ("training_prevalence", "train_prevalence"),
            74069 / 951722,
        ),
    )
    training_prevalence = float(training_prevalence)

    with rasterio.open(REFERENCE_RASTER) as reference:
        height, width = reference.height, reference.width
        profile = reference.profile.copy()
        crs = reference.crs

    print("\n" + "=" * 78)
    print("CLIMATEADAPT V3 – FINAL LOCKED GEOGRAPHIC TEST")
    print("=" * 78)
    print("Test regions: Delaware Open Coast + Oananga Bay")
    print(f"Test tiles: {len(test_manifest)}")
    print(f"Reference raster: {width:,} x {height:,} | CRS: {crs}")
    print(f"Device: {device}")
    print(f"U-Net checkpoint epoch: {checkpoint_epoch}")
    if checkpoint_ap is not None:
        print(f"U-Net validation PR-AUC: {float(checkpoint_ap):.4f}")
    print(f"Frozen U-Net operational threshold: {unet_threshold:.3f}")
    print(f"Frozen U-Net Dice comparison threshold: {dice_threshold:.3f}")
    print(f"Frozen logistic threshold: {logistic_threshold:.3f}")
    print("No training or threshold selection will occur.")
    print("Running final locked-test inference...")

    all_indices, all_truth, all_areas = [], [], []
    all_unet, all_logistic = [], []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for sequence, (_, row) in enumerate(test_manifest.iterrows(), start=1):
            tile_path = resolve_tile_path(str(row["file"]))
            image, target, valid, area = load_tile(tile_path, input_channels)
            tile_height, tile_width = target.shape
            row_start = int(row["row"])
            column_start = int(row["column"])
            row_stop = min(row_start + tile_height, height)
            column_stop = min(column_start + tile_width, width)
            local_height = row_stop - row_start
            local_width = column_stop - column_start
            if local_height <= 0 or local_width <= 0:
                continue
            image = image[:, :local_height, :local_width]
            target = target[:local_height, :local_width]
            valid = valid[:local_height, :local_width]
            area = area[:local_height, :local_width]

            # Some tile writers may omit the area array. The manifest area column
            # is an accepted fallback, but unknown test areas are never guessed.
            if not np.any(area) and "area_id" in row.index:
                area.fill(int(row["area_id"]))
            elif not np.any(area) and "area_zone_id" in row.index:
                area.fill(int(row["area_zone_id"]))

            tensor = torch.from_numpy(image[None]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(tensor)
            unet_probability = torch.sigmoid(logits).squeeze().float().cpu().numpy()

            flat_valid = valid.ravel()
            feature_matrix = np.moveaxis(image, 0, -1).reshape(-1, input_channels)[flat_valid]
            expected_features = getattr(logistic_model, "n_features_in_", input_channels)
            if int(expected_features) != input_channels:
                raise ValueError(
                    f"Logistic model expects {expected_features} features; V3 tiles contain "
                    f"{input_channels}. Recreate the V3 baseline artifact with Script 41."
                )
            logistic_probability = logistic_model.predict_proba(feature_matrix)[:, 1]

            local_rows, local_columns = np.indices((local_height, local_width))
            global_indices = (
                (local_rows + row_start) * width + local_columns + column_start
            ).ravel()[flat_valid]
            all_indices.append(global_indices.astype(np.int64))
            all_truth.append(target.ravel()[flat_valid].astype(np.uint8))
            all_areas.append(area.ravel()[flat_valid].astype(np.uint8))
            all_unet.append(unet_probability[:local_height, :local_width].ravel()[flat_valid])
            all_logistic.append(logistic_probability.astype(np.float32))
            if sequence % 50 == 0 or sequence == len(test_manifest):
                print(f"Processed test tile {sequence}/{len(test_manifest)}")

    merged = merge_overlaps(
        np.concatenate(all_indices),
        np.concatenate(all_truth),
        np.concatenate(all_areas),
        np.concatenate(all_unet),
        np.concatenate(all_logistic),
    )
    indices, truth, areas, unet_probability, logistic_probability, duplicates = merged
    observed_areas = set(np.unique(areas).tolist())
    if observed_areas != EXPECTED_TEST_AREAS:
        raise RuntimeError(
            f"Expected test area IDs {sorted(EXPECTED_TEST_AREAS)}, observed "
            f"{sorted(observed_areas)}. Refusing to report an ambiguous final test."
        )

    trivial_probability = np.full(truth.shape, training_prevalence, dtype=np.float32)
    pooled_records = [
        metric_record(truth, trivial_probability, 0.5, "Trivial prevalence", "Pooled test"),
        metric_record(truth, logistic_probability, logistic_threshold, "Logistic regression", "Pooled test"),
        metric_record(truth, unet_probability, unet_threshold, "U-Net operational", "Pooled test"),
        metric_record(truth, unet_probability, dice_threshold, "U-Net Dice comparison", "Pooled test"),
    ]
    area_records = []
    for area_id, area_name in AREA_NAMES.items():
        selector = areas == area_id
        for probability, threshold, name in (
            (trivial_probability, 0.5, "Trivial prevalence"),
            (logistic_probability, logistic_threshold, "Logistic regression"),
            (unet_probability, unet_threshold, "U-Net operational"),
            (unet_probability, dice_threshold, "U-Net Dice comparison"),
        ):
            record = metric_record(
                truth[selector], probability[selector], threshold, name, area_name
            )
            record["area_id"] = area_id
            area_records.append(record)

    pd.DataFrame(pooled_records).to_csv(POOLED_CSV, index=False)
    pd.DataFrame(area_records).to_csv(AREA_CSV, index=False)

    print("Writing final test rasters...")
    save_indexed_raster(UNET_PROBABILITY_RASTER, indices, unet_probability, profile, "float32", -9999.0)
    save_indexed_raster(UNET_BINARY_RASTER, indices, (unet_probability >= unet_threshold), profile, "uint8", 255)
    save_indexed_raster(LOGISTIC_PROBABILITY_RASTER, indices, logistic_probability, profile, "float32", -9999.0)
    save_indexed_raster(LOGISTIC_BINARY_RASTER, indices, (logistic_probability >= logistic_threshold), profile, "uint8", 255)
    save_indexed_raster(TARGET_RASTER, indices, truth, profile, "uint8", 255)
    unet_error = np.zeros(truth.shape, dtype=np.uint8)
    unet_prediction = unet_probability >= unet_threshold
    unet_error[(truth == 0) & unet_prediction] = 1  # false positive
    unet_error[(truth == 1) & ~unet_prediction] = 2  # false negative
    unet_error[(truth == 1) & unet_prediction] = 3  # true positive
    save_indexed_raster(UNET_ERROR_RASTER, indices, unet_error, profile, "uint8", 255)

    create_map_figure(
        indices, truth, areas, unet_probability, logistic_probability,
        unet_threshold, logistic_threshold, width,
    )
    create_curve_figure(truth, unet_probability, logistic_probability)

    unet_record = next(item for item in pooled_records if item["model"] == "U-Net operational")
    logistic_record = next(item for item in pooled_records if item["model"] == "Logistic regression")
    runtime = time.perf_counter() - started
    summary = {
        "experiment": "ClimateAdapt V3 final locked geographic test",
        "test_opened": True,
        "no_test_driven_tuning_permitted": True,
        "test_regions": list(AREA_NAMES.values()),
        "test_area_ids": AREA_NAMES,
        "test_tiles": int(len(test_manifest)),
        "unique_test_pixels": int(truth.size),
        "overlapping_duplicate_predictions_merged": duplicates,
        "positive_test_pixels": int(np.sum(truth)),
        "test_prevalence": float(np.mean(truth)),
        "unet": {
            "checkpoint": str(UNET_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "checkpoint_epoch": checkpoint_epoch,
            "validation_pr_auc": float(checkpoint_ap) if checkpoint_ap is not None else None,
            "operational_threshold": unet_threshold,
            "dice_comparison_threshold": dice_threshold,
            "pooled_operational_metrics": unet_record,
        },
        "logistic": {
            "model": str(LOGISTIC_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "validation_selected_threshold": logistic_threshold,
            "pooled_metrics": logistic_record,
        },
        "trivial_training_prevalence": training_prevalence,
        "primary_model_predeclared_from_validation": "U-Net V3",
        "test_comparison_is_evaluation_not_model_selection": True,
        "per_area_metrics": area_records,
        "processing_time_seconds": runtime,
        "artifacts": {
            "pooled_csv": str(POOLED_CSV.relative_to(PROJECT_ROOT)),
            "area_csv": str(AREA_CSV.relative_to(PROJECT_ROOT)),
            "map_figure": str(MAP_FIGURE.relative_to(PROJECT_ROOT)),
            "curves_figure": str(CURVES_FIGURE.relative_to(PROJECT_ROOT)),
        },
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("V3 FINAL LOCKED TEST COMPLETED")
    print("=" * 78)
    print(f"Unique test pixels: {truth.size:,}")
    print(f"Duplicate predictions merged: {duplicates:,}")
    print(f"Positive test pixels: {np.sum(truth):,} ({100*np.mean(truth):.2f}%)")
    print("-" * 78)
    for record in pooled_records:
        print(
            f"{record['model']:<25} AP={record['average_precision']:.4f} "
            f"F2={record['f2_score']:.4f} Dice={record['dice']:.4f} "
            f"P={record['precision']:.4f} R={record['recall']:.4f}"
        )
    print("-" * 78)
    print(f"Pooled metrics:  {POOLED_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Per-area metrics:{AREA_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Summary:         {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Evaluation map:  {MAP_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Curves:          {CURVES_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Processing time: {runtime:.2f} seconds")
    print("=" * 78)
    print("The locked test has now been opened. Do not change model weights,")
    print("features, splits or thresholds using these results.")


if __name__ == "__main__":
    main()
