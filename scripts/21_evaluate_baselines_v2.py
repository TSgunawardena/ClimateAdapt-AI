"""
ClimateAdapt AI
Experiment 2 baseline evaluation

Creates:
1. A trivial training-prevalence / majority-class baseline.
2. An interpretable pixel-level logistic-regression baseline.
3. A fair comparison with the frozen U-Net V2 test results.

Leakage controls:
- Training uses only Nelson CBD / Nelson Haven.
- Threshold selection uses only Rocks Road validation.
- Tahunanui is accessed only after the threshold is frozen.
- Duplicate pixels from overlapping tiles are merged using raster coordinates.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RASTER_DIR = PROJECT_ROOT / "data" / "processed" / "combined_rasters"
TILE_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "tiles" / "tile_manifest.csv"
)

ELEVATION_PATH = RASTER_DIR / "elevation_5m.tif"
SLOPE_PATH = RASTER_DIR / "slope_degrees_5m.tif"
LAND_COVER_PATH = RASTER_DIR / "land_cover_class_5m.tif"
COAST_DISTANCE_PATH = RASTER_DIR / "distance_to_coast_5m.tif"
FLOOD_TARGET_PATH = RASTER_DIR / "flood_target_5m.tif"
VALID_MASK_PATH = RASTER_DIR / "valid_land_mask_5m.tif"
SPLIT_MASK_PATH = RASTER_DIR / "split_mask_5m.tif"

UNET_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "test_v2"
    / "tahunanui_test_metrics_v2.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines_v2"
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

METRICS_JSON_PATH = OUTPUT_DIR / "baseline_comparison_v2.json"
METRICS_CSV_PATH = OUTPUT_DIR / "baseline_comparison_v2.csv"
THRESHOLD_CSV_PATH = OUTPUT_DIR / "logistic_threshold_search_v2.csv"
COEFFICIENTS_CSV_PATH = OUTPUT_DIR / "logistic_coefficients_v2.csv"
SELECTED_THRESHOLD_PATH = OUTPUT_DIR / "selected_logistic_threshold_v2.json"

MODEL_PATH = MODEL_DIR / "logistic_regression_baseline_v2.joblib"

COMPARISON_FIGURE_PATH = (
    FIGURE_DIR / "baseline_vs_unet_v2.png"
)
THRESHOLD_FIGURE_PATH = (
    FIGURE_DIR / "logistic_validation_threshold_v2.png"
)

TILE_SIZE = 128

SPLIT_IDS = {
    "train": 1,
    "validation": 2,
    "test": 3,
}

RANDOM_SEED = 42


# ============================================================================
# HELPERS
# ============================================================================

def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required raster not found: {path}")

    with rasterio.open(path) as source:
        array = source.read(1)
        metadata = {
            "shape": (source.height, source.width),
            "crs": str(source.crs),
            "transform": tuple(source.transform),
            "nodata": source.nodata,
        }

    return array, metadata


def normalise_split_name(value: str) -> str | None:
    value = str(value).strip().lower()

    aliases = {
        "train": "train",
        "training": "train",
        "validation": "validation",
        "validate": "validation",
        "val": "validation",
        "test": "test",
        "testing": "test",
    }

    return aliases.get(value)


def create_tile_coverage_masks(
    manifest: pd.DataFrame,
    raster_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    required_columns = {"split", "row", "column"}

    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            "Tile manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    coverage = {
        split_name: np.zeros(raster_shape, dtype=bool)
        for split_name in SPLIT_IDS
    }

    accepted_rows = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    height, width = raster_shape

    for manifest_row in manifest.itertuples(index=False):
        split_name = normalise_split_name(getattr(manifest_row, "split"))

        # Demonstration rows and unknown split names are deliberately ignored.
        if split_name not in coverage:
            continue

        row_start = int(getattr(manifest_row, "row"))
        column_start = int(getattr(manifest_row, "column"))

        row_end = min(row_start + TILE_SIZE, height)
        column_end = min(column_start + TILE_SIZE, width)

        if (
            row_start < 0
            or column_start < 0
            or row_start >= height
            or column_start >= width
        ):
            raise ValueError(
                "Manifest contains an invalid tile origin: "
                f"split={split_name}, row={row_start}, "
                f"column={column_start}"
            )

        coverage[split_name][
            row_start:row_end,
            column_start:column_end,
        ] = True

        accepted_rows[split_name] += 1

    print("\nManifest tiles used")
    print("-" * 72)
    for split_name in ("train", "validation", "test"):
        print(
            f"{split_name.title():12s}: "
            f"{accepted_rows[split_name]:4d} tiles"
        )

    return coverage


def extract_unique_pixels(
    mask: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    coast_distance: np.ndarray,
    land_cover: np.ndarray,
    flood_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_indices = np.flatnonzero(mask)

    if flat_indices.size == 0:
        raise ValueError("No valid pixels were found for this split.")

    elevation_values = elevation.ravel()[flat_indices].astype(np.float32)
    slope_values = slope.ravel()[flat_indices].astype(np.float32)
    distance_values = coast_distance.ravel()[flat_indices].astype(np.float32)
    land_cover_values = land_cover.ravel()[flat_indices].astype(np.float32)
    target_values = flood_target.ravel()[flat_indices].astype(np.uint8)

    # Match the U-Net feature engineering: coastline distance is log transformed.
    distance_values = np.log1p(np.maximum(distance_values, 0.0))

    features = np.column_stack(
        [
            elevation_values,
            slope_values,
            distance_values,
            land_cover_values,
        ]
    ).astype(np.float32)

    return features, target_values


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = 10,
) -> float:
    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    result = 0.0

    for bin_index in range(number_of_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == number_of_bins - 1:
            in_bin = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            in_bin = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        number_in_bin = int(in_bin.sum())

        if number_in_bin == 0:
            continue

        mean_confidence = float(probabilities[in_bin].mean())
        observed_frequency = float(targets[in_bin].mean())
        bin_weight = number_in_bin / len(targets)

        result += (
            abs(mean_confidence - observed_frequency)
            * bin_weight
        )

    return float(result)


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.uint8)

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    specificity_denominator = true_negative + false_positive
    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator > 0
        else 0.0
    )

    false_negative_denominator = true_positive + false_negative
    false_negative_rate = (
        false_negative / false_negative_denominator
        if false_negative_denominator > 0
        else 0.0
    )

    dice_denominator = (
        2 * true_positive + false_positive + false_negative
    )
    dice = (
        (2 * true_positive) / dice_denominator
        if dice_denominator > 0
        else 0.0
    )

    iou_denominator = true_positive + false_positive + false_negative
    iou = (
        true_positive / iou_denominator
        if iou_denominator > 0
        else 0.0
    )

    if np.unique(targets).size == 2:
        roc_auc = float(roc_auc_score(targets, probabilities))
    else:
        roc_auc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "precision": float(
            precision_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "false_negative_rate": float(false_negative_rate),
        "f2_score": float(
            fbeta_score(
                targets,
                predictions,
                beta=2,
                zero_division=0,
            )
        ),
        "dice": float(dice),
        "iou": float(iou),
        "roc_auc": roc_auc,
        "average_precision": float(
            average_precision_score(targets, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(targets, probabilities)
        ),
        "calibration_error": float(
            expected_calibration_error(
                targets,
                probabilities,
            )
        ),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "evaluated_pixels": int(len(targets)),
        "positive_pixels": int(targets.sum()),
        "positive_percentage": float(targets.mean() * 100.0),
    }


def select_operational_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows = []

    # Fine-grained threshold search using validation data only.
    for threshold in np.round(
        np.arange(0.01, 1.00, 0.01),
        2,
    ):
        metrics = calculate_metrics(
            targets,
            probabilities,
            float(threshold),
        )

        rows.append(
            {
                "threshold": metrics["threshold"],
                "f2_score": metrics["f2_score"],
                "dice": metrics["dice"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "false_negative_rate": metrics[
                    "false_negative_rate"
                ],
            }
        )

    results = pd.DataFrame(rows)

    # Primary criterion: maximum F2.
    # Tie-breaking: higher recall, then lower threshold.
    selected_row = results.sort_values(
        by=["f2_score", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]

    selected_threshold = float(selected_row["threshold"])

    return selected_threshold, results


def build_logistic_pipeline() -> Pipeline:
    numeric_columns = [0, 1, 2]
    categorical_columns = [3]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            ),
            (
                "land_cover",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        solver="saga",
        # penalty="l2",
        class_weight="balanced",
        max_iter=1500,
        tol=1e-4,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def flatten_numeric_json(
    value,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], float]]:
    flattened = []

    if isinstance(value, dict):
        for key, child_value in value.items():
            flattened.extend(
                flatten_numeric_json(
                    child_value,
                    path + (str(key),),
                )
            )

    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened.append((path, float(value)))

    return flattened


def load_unet_operational_metrics(
    path: Path,
) -> dict[str, float]:
    if not path.exists():
        print(
            "\nWARNING: U-Net metrics JSON was not found. "
            "The baseline will still be evaluated."
        )
        return {}

    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    numeric_values = flatten_numeric_json(content)

    aliases = {
        "threshold": {
            "threshold",
            "operationalthreshold",
        },
        "accuracy": {
            "accuracy",
        },
        "balanced_accuracy": {
            "balancedaccuracy",
        },
        "precision": {
            "precision",
        },
        "recall": {
            "recall",
            "sensitivity",
            "recallsensitivity",
        },
        "specificity": {
            "specificity",
        },
        "false_negative_rate": {
            "falsenegativerate",
            "fnr",
        },
        "f2_score": {
            "f2",
            "f2score",
        },
        "dice": {
            "dice",
            "dicescore",
        },
        "iou": {
            "iou",
            "intersectionoverunion",
        },
        "roc_auc": {
            "rocauc",
            "aucroc",
        },
        "average_precision": {
            "averageprecision",
            "prauc",
            "precisionrecallauc",
        },
        "brier_score": {
            "brierscore",
        },
        "calibration_error": {
            "calibrationerror",
            "expectedcalibrationerror",
            "ece",
        },
    }

    extracted = {}

    for output_name, possible_names in aliases.items():
        candidates = []

        for numeric_path, numeric_value in numeric_values:
            if not numeric_path:
                continue

            leaf_name = normalise_key(numeric_path[-1])
            full_path = normalise_key(" ".join(numeric_path))

            if leaf_name not in possible_names:
                continue

            score = 0

            if "operational" in full_path:
                score += 20
            if "primary" in full_path:
                score += 15
            if "test" in full_path:
                score += 5
            if "secondary" in full_path:
                score -= 20
            if "comparison" in full_path:
                score -= 15
            if "validation" in full_path:
                score -= 10

            candidates.append(
                (
                    score,
                    len(numeric_path),
                    numeric_value,
                    ".".join(numeric_path),
                )
            )

        if candidates:
            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                ),
                reverse=True,
            )
            extracted[output_name] = float(candidates[0][2])

    return extracted


def save_logistic_coefficients(model: Pipeline) -> None:
    preprocessing = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessing.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "direction": np.where(
                coefficients >= 0,
                "increases predicted susceptibility",
                "decreases predicted susceptibility",
            ),
        }
    ).sort_values(
        by="absolute_coefficient",
        ascending=False,
    )

    coefficient_table.to_csv(
        COEFFICIENTS_CSV_PATH,
        index=False,
    )


def create_threshold_figure(
    threshold_results: pd.DataFrame,
    selected_threshold: float,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        threshold_results["threshold"],
        threshold_results["f2_score"],
        label="F2 score",
        linewidth=2.4,
    )
    axis.plot(
        threshold_results["threshold"],
        threshold_results["dice"],
        label="Dice",
        linewidth=2.0,
    )
    axis.plot(
        threshold_results["threshold"],
        threshold_results["precision"],
        label="Precision",
        linewidth=1.8,
    )
    axis.plot(
        threshold_results["threshold"],
        threshold_results["recall"],
        label="Recall",
        linewidth=1.8,
    )

    axis.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Selected threshold = {selected_threshold:.2f}",
    )

    axis.set_title(
        "Logistic Baseline – Rocks Road Validation Threshold"
    )
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Metric value")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(
        THRESHOLD_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_comparison_figure(
    result_rows: list[dict],
) -> None:
    comparison_metrics = [
        "balanced_accuracy",
        "precision",
        "recall",
        "f2_score",
        "dice",
        "iou",
        "roc_auc",
        "average_precision",
    ]

    comparison_labels = [
        "Balanced\naccuracy",
        "Precision",
        "Recall",
        "F2",
        "Dice",
        "IoU",
        "ROC-AUC",
        "Average\nprecision",
    ]

    valid_rows = []

    for row in result_rows:
        available_count = sum(
            metric_name in row
            and pd.notna(row[metric_name])
            for metric_name in comparison_metrics
        )

        if available_count >= 4:
            valid_rows.append(row)

    if not valid_rows:
        print(
            "WARNING: Not enough metrics were available "
            "to create the comparison figure."
        )
        return

    x_positions = np.arange(len(comparison_metrics))
    number_of_models = len(valid_rows)
    group_width = 0.82
    bar_width = group_width / number_of_models

    figure, axis = plt.subplots(figsize=(14, 7))

    colours = [
        "#8c8c8c",
        "#e69f00",
        "#0072b2",
        "#009e73",
    ]

    for model_index, row in enumerate(valid_rows):
        offset = (
            model_index - (number_of_models - 1) / 2
        ) * bar_width

        values = [
            row.get(metric_name, np.nan)
            for metric_name in comparison_metrics
        ]

        axis.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=row["model"],
            color=colours[model_index % len(colours)],
            edgecolor="black",
            linewidth=0.4,
        )

    axis.set_title(
        "ClimateAdapt AI – Independent Tahunanui Model Comparison"
    )
    axis.set_ylabel("Metric value")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(comparison_labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", ncol=number_of_models)

    figure.tight_layout()
    figure.savefig(
        COMPARISON_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def serialisable_metrics(metrics: dict) -> dict:
    result = {}

    for key, value in metrics.items():
        if isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        else:
            result[key] = value

    return result


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main() -> None:
    ensure_directories()
    start_time = time.perf_counter()

    print("\n" + "=" * 78)
    print("CLIMATEADAPT EXPERIMENT 2 – BASELINE MODEL EVALUATION")
    print("=" * 78)
    print("Training region:   Nelson CBD + Nelson Haven")
    print("Validation region: Rocks Road")
    print("Test region:       Tahunanui")
    print("Baseline tribe:    Symbolist / statistical linear classifier")
    print("Test threshold is selected using validation data only.")

    print("\nLoading aligned rasters...")

    elevation, reference_metadata = load_raster(ELEVATION_PATH)
    slope, slope_metadata = load_raster(SLOPE_PATH)
    land_cover, land_cover_metadata = load_raster(LAND_COVER_PATH)
    coast_distance, distance_metadata = load_raster(
        COAST_DISTANCE_PATH
    )
    flood_target, target_metadata = load_raster(FLOOD_TARGET_PATH)
    valid_land, valid_metadata = load_raster(VALID_MASK_PATH)
    split_mask, split_metadata = load_raster(SPLIT_MASK_PATH)

    expected_shape = reference_metadata["shape"]

    metadata_items = {
        "slope": slope_metadata,
        "land cover": land_cover_metadata,
        "coast distance": distance_metadata,
        "flood target": target_metadata,
        "valid mask": valid_metadata,
        "split mask": split_metadata,
    }

    for dataset_name, metadata in metadata_items.items():
        if metadata["shape"] != expected_shape:
            raise ValueError(
                f"{dataset_name} shape {metadata['shape']} "
                f"does not match reference shape {expected_shape}."
            )

    print(
        f"Raster size: {expected_shape[1]:,} columns "
        f"x {expected_shape[0]:,} rows"
    )
    print(f"CRS: {reference_metadata['crs']}")

    manifest = pd.read_csv(TILE_MANIFEST_PATH)

    coverage_masks = create_tile_coverage_masks(
        manifest,
        expected_shape,
    )

    finite_feature_mask = (
        np.isfinite(elevation)
        & np.isfinite(slope)
        & np.isfinite(coast_distance)
        & np.isfinite(land_cover)
    )

    split_pixel_masks = {}

    for split_name, split_id in SPLIT_IDS.items():
        split_pixel_masks[split_name] = (
            coverage_masks[split_name]
            & (valid_land > 0)
            & (split_mask == split_id)
            & finite_feature_mask
        )

    print("\nExtracting unique pixels...")
    print("-" * 72)

    train_features, train_targets = extract_unique_pixels(
        split_pixel_masks["train"],
        elevation,
        slope,
        coast_distance,
        land_cover,
        flood_target,
    )

    validation_features, validation_targets = extract_unique_pixels(
        split_pixel_masks["validation"],
        elevation,
        slope,
        coast_distance,
        land_cover,
        flood_target,
    )

    test_features, test_targets = extract_unique_pixels(
        split_pixel_masks["test"],
        elevation,
        slope,
        coast_distance,
        land_cover,
        flood_target,
    )

    print(
        f"Training:   {len(train_targets):,} unique pixels | "
        f"positive={train_targets.sum():,} "
        f"({train_targets.mean() * 100:.3f}%)"
    )
    print(
        f"Validation: {len(validation_targets):,} unique pixels | "
        f"positive={validation_targets.sum():,} "
        f"({validation_targets.mean() * 100:.3f}%)"
    )
    print(
        f"Testing:    {len(test_targets):,} unique pixels | "
        f"positive={test_targets.sum():,} "
        f"({test_targets.mean() * 100:.3f}%)"
    )

    # ----------------------------------------------------------------------
    # Trivial baseline
    # ----------------------------------------------------------------------

    print("\nEvaluating trivial training-prevalence baseline...")

    training_prevalence = float(train_targets.mean())

    trivial_validation_probabilities = np.full(
        len(validation_targets),
        training_prevalence,
        dtype=np.float32,
    )
    trivial_test_probabilities = np.full(
        len(test_targets),
        training_prevalence,
        dtype=np.float32,
    )

    # At 0.5 this becomes the majority negative baseline.
    trivial_threshold = 0.50

    trivial_validation_metrics = calculate_metrics(
        validation_targets,
        trivial_validation_probabilities,
        trivial_threshold,
    )
    trivial_test_metrics = calculate_metrics(
        test_targets,
        trivial_test_probabilities,
        trivial_threshold,
    )

    # ----------------------------------------------------------------------
    # Logistic regression baseline
    # ----------------------------------------------------------------------

    print("Training logistic-regression baseline...")
    logistic_training_start = time.perf_counter()

    logistic_model = build_logistic_pipeline()
    logistic_model.fit(train_features, train_targets)

    logistic_training_seconds = (
        time.perf_counter() - logistic_training_start
    )

    print(
        f"Logistic training time: "
        f"{logistic_training_seconds:.2f} seconds"
    )

    logistic_validation_probabilities = logistic_model.predict_proba(
        validation_features
    )[:, 1]

    selected_logistic_threshold, threshold_results = (
        select_operational_threshold(
            validation_targets,
            logistic_validation_probabilities,
        )
    )

    logistic_validation_metrics = calculate_metrics(
        validation_targets,
        logistic_validation_probabilities,
        selected_logistic_threshold,
    )

    print(
        f"Frozen Rocks Road F2 threshold: "
        f"{selected_logistic_threshold:.2f}"
    )
    print(
        f"Validation F2: "
        f"{logistic_validation_metrics['f2_score']:.4f}"
    )
    print(
        f"Validation Dice: "
        f"{logistic_validation_metrics['dice']:.4f}"
    )

    print(
        "\nRunning the logistic baseline once on "
        "independent Tahunanui test pixels..."
    )

    logistic_test_probabilities = logistic_model.predict_proba(
        test_features
    )[:, 1]

    logistic_test_metrics = calculate_metrics(
        test_targets,
        logistic_test_probabilities,
        selected_logistic_threshold,
    )

    # ----------------------------------------------------------------------
    # Existing U-Net metrics
    # ----------------------------------------------------------------------

    unet_metrics = load_unet_operational_metrics(
        UNET_METRICS_PATH
    )

    if unet_metrics:
        print(
            f"Loaded {len(unet_metrics)} U-Net metric fields "
            f"from {UNET_METRICS_PATH.relative_to(PROJECT_ROOT)}"
        )
    else:
        print(
            "WARNING: U-Net metrics could not be extracted "
            "from the JSON file."
        )

    # ----------------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------------

    joblib.dump(logistic_model, MODEL_PATH)
    threshold_results.to_csv(
        THRESHOLD_CSV_PATH,
        index=False,
    )
    save_logistic_coefficients(logistic_model)

    with SELECTED_THRESHOLD_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "model": "pixel-level logistic regression baseline",
                "threshold": selected_logistic_threshold,
                "selection_region": "Rocks Road validation",
                "selection_metric": "F2 score",
                "test_region": "Tahunanui",
                "test_data_used_during_selection": False,
            },
            file,
            indent=2,
        )

    comparison_rows = [
        {
            "model": "Trivial majority baseline",
            **serialisable_metrics(trivial_test_metrics),
        },
        {
            "model": "Logistic regression baseline",
            **serialisable_metrics(logistic_test_metrics),
        },
    ]

    if unet_metrics:
        comparison_rows.append(
            {
                "model": "Compact U-Net V2",
                **serialisable_metrics(unet_metrics),
            }
        )

    pd.DataFrame(comparison_rows).to_csv(
        METRICS_CSV_PATH,
        index=False,
    )

    elapsed_seconds = time.perf_counter() - start_time

    output_document = {
        "experiment": "ClimateAdapt AI Experiment 2 baseline comparison",
        "random_seed": RANDOM_SEED,
        "features": [
            "elevation",
            "slope",
            "log1p distance to coastline",
            "one-hot encoded LCDB 2023 land-cover class",
        ],
        "spatial_split": {
            "training": "Nelson CBD + Nelson Haven",
            "validation": "Rocks Road",
            "testing": "Tahunanui",
        },
        "leakage_controls": [
            "Geographically separated train, validation and test regions",
            "Training-only model fitting",
            "Validation-only threshold selection",
            "Overlapping tile predictions merged by unique raster pixel",
            "Tahunanui not used for fitting or threshold selection",
        ],
        "training_pixels": int(len(train_targets)),
        "validation_pixels": int(len(validation_targets)),
        "test_pixels": int(len(test_targets)),
        "training_prevalence": training_prevalence,
        "trivial_validation_metrics": serialisable_metrics(
            trivial_validation_metrics
        ),
        "trivial_test_metrics": serialisable_metrics(
            trivial_test_metrics
        ),
        "logistic_validation_metrics": serialisable_metrics(
            logistic_validation_metrics
        ),
        "logistic_test_metrics": serialisable_metrics(
            logistic_test_metrics
        ),
        "unet_v2_test_metrics": serialisable_metrics(
            unet_metrics
        ),
        "logistic_training_seconds": logistic_training_seconds,
        "total_processing_seconds": elapsed_seconds,
        "interpretation_limitations": [
            "The logistic model predicts pixels independently and does not learn spatial context.",
            "Logistic coefficients describe conditional association, not causation.",
            "One-hot land-cover coefficients depend on the omitted reference category.",
            "Performance on Tahunanui does not prove transferability outside Nelson.",
        ],
    }

    with METRICS_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_document,
            file,
            indent=2,
            allow_nan=False,
        )

    create_threshold_figure(
        threshold_results,
        selected_logistic_threshold,
    )
    create_comparison_figure(comparison_rows)

    # ----------------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("INDEPENDENT TAHUNANUI BASELINE RESULTS")
    print("=" * 78)

    print("\nTRIVIAL MAJORITY BASELINE")
    print(f"Threshold:          {trivial_threshold:.2f}")
    print(
        f"Balanced accuracy:  "
        f"{trivial_test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Precision:          "
        f"{trivial_test_metrics['precision']:.4f}"
    )
    print(
        f"Recall:             "
        f"{trivial_test_metrics['recall']:.4f}"
    )
    print(
        f"F2 score:           "
        f"{trivial_test_metrics['f2_score']:.4f}"
    )
    print(
        f"Dice:               "
        f"{trivial_test_metrics['dice']:.4f}"
    )
    print(
        f"ROC-AUC:            "
        f"{trivial_test_metrics['roc_auc']:.4f}"
    )
    print(
        f"Average precision:  "
        f"{trivial_test_metrics['average_precision']:.4f}"
    )

    print("\nLOGISTIC REGRESSION BASELINE")
    print(
        f"Frozen threshold:   "
        f"{selected_logistic_threshold:.2f}"
    )
    print(
        f"Accuracy:           "
        f"{logistic_test_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy:  "
        f"{logistic_test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Precision:          "
        f"{logistic_test_metrics['precision']:.4f}"
    )
    print(
        f"Recall:             "
        f"{logistic_test_metrics['recall']:.4f}"
    )
    print(
        f"F2 score:           "
        f"{logistic_test_metrics['f2_score']:.4f}"
    )
    print(
        f"Dice:               "
        f"{logistic_test_metrics['dice']:.4f}"
    )
    print(
        f"IoU:                "
        f"{logistic_test_metrics['iou']:.4f}"
    )
    print(
        f"ROC-AUC:            "
        f"{logistic_test_metrics['roc_auc']:.4f}"
    )
    print(
        f"Average precision:  "
        f"{logistic_test_metrics['average_precision']:.4f}"
    )
    print(
        f"Brier score:        "
        f"{logistic_test_metrics['brier_score']:.4f}"
    )

    if unet_metrics:
        print("\nFROZEN COMPACT U-NET V2")
        for metric_name in (
            "balanced_accuracy",
            "precision",
            "recall",
            "f2_score",
            "dice",
            "iou",
            "roc_auc",
            "average_precision",
            "brier_score",
        ):
            if metric_name in unet_metrics:
                print(
                    f"{metric_name.replace('_', ' ').title():20s}"
                    f"{unet_metrics[metric_name]:.4f}"
                )

    print("\n" + "-" * 78)
    print(
        f"Metrics JSON:       "
        f"{METRICS_JSON_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Metrics CSV:        "
        f"{METRICS_CSV_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Threshold results:  "
        f"{THRESHOLD_CSV_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Coefficients:       "
        f"{COEFFICIENTS_CSV_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Saved model:        "
        f"{MODEL_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Comparison figure:  "
        f"{COMPARISON_FIGURE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Threshold figure:   "
        f"{THRESHOLD_FIGURE_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(f"Processing time:    {elapsed_seconds:.2f} seconds")
    print("=" * 78)
    print(
        "The Tahunanui test area was not used for training "
        "or threshold selection."
    )


if __name__ == "__main__":
    main()