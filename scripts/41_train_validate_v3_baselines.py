"""Train and validate ClimateAdapt V3 trivial and logistic baselines.

Only train and validation tile files are opened. Overlapping tile pixels are
deduplicated using their full-raster coordinates. The logistic operational
threshold is selected on validation F2. The locked test folder is not opened.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TILE_ROOT = PROJECT_ROOT / "data" / "processed" / "v3" / "tiles"
MANIFEST_PATH = TILE_ROOT / "v3_tile_manifest.csv"
TILE_METADATA_PATH = TILE_ROOT / "v3_tile_metadata.json"
REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "v3" / "v3_split_mask_5m.tif"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "baselines_v3"
MODEL_ROOT = PROJECT_ROOT / "models" / "v3"
FIGURE_ROOT = PROJECT_ROOT / "evidence" / "figures" / "v3"

MODEL_PATH = MODEL_ROOT / "logistic_regression_v3.joblib"
SUMMARY_PATH = OUTPUT_ROOT / "baseline_validation_summary_v3.json"
COMPARISON_PATH = OUTPUT_ROOT / "baseline_validation_comparison_v3.csv"
THRESHOLD_PATH = OUTPUT_ROOT / "logistic_validation_threshold_search_v3.csv"
COEFFICIENT_PATH = OUTPUT_ROOT / "logistic_coefficients_v3.csv"
SELECTED_THRESHOLD_PATH = OUTPUT_ROOT / "logistic_selected_threshold_v3.json"
FIGURE_PATH = FIGURE_ROOT / "v3_baseline_validation.png"

RANDOM_SEED = 8532


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_unique_pixels(manifest: pd.DataFrame, split_name: str, raster_width: int):
    records = manifest[manifest["split"] == split_name]
    if records.empty:
        fail(f"No manifest records exist for split '{split_name}'.")

    indices_parts = []
    x_parts = []
    y_parts = []
    opened = 0
    for row in records.itertuples(index=False):
        tile_path = TILE_ROOT / Path(row.file)
        if not tile_path.exists():
            fail(f"Missing {split_name} tile: {tile_path}")
        with np.load(tile_path) as tile:
            x = tile["x"]
            y = tile["y"]
            mask = tile["valid_mask"].astype(bool)
            tile_row = int(tile["row"])
            tile_column = int(tile["column"])
        local_rows, local_columns = np.nonzero(mask)
        global_index = (
            (tile_row + local_rows).astype(np.int64) * raster_width
            + (tile_column + local_columns).astype(np.int64)
        )
        indices_parts.append(global_index)
        x_parts.append(x[:, mask].T.astype(np.float32, copy=False))
        y_parts.append(y[mask].astype(np.uint8, copy=False))
        opened += 1
        if opened % 100 == 0 or opened == len(records):
            print(f"  {split_name}: loaded {opened}/{len(records)} tiles")

    indices = np.concatenate(indices_parts)
    features = np.concatenate(x_parts, axis=0)
    target = np.concatenate(y_parts)
    _, first_positions = np.unique(indices, return_index=True)
    first_positions.sort()
    duplicates = int(len(indices) - len(first_positions))
    return (
        features[first_positions],
        target[first_positions],
        int(len(first_positions)),
        duplicates,
    )


def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    dice = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "false_negative_rate": float(1.0 - recall),
        "f2_score": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "dice": float(dice),
        "iou": float(jaccard_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def threshold_search(y_true, probabilities) -> pd.DataFrame:
    thresholds = np.unique(
        np.concatenate(
            [
                np.arange(0.001, 0.010, 0.001),
                np.arange(0.01, 1.00, 0.01),
            ]
        )
    )
    records = []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        records.append(
            {
                "threshold": float(threshold),
                "f2_score": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
                "dice": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
                "false_negative_rate": float(1.0 - recall),
            }
        )
    return pd.DataFrame(records)


def feature_names(tile_metadata: dict) -> list[str]:
    names = ["elevation", "slope", "log1p_distance_to_coast"]
    groups = tile_metadata["land_cover_semantic_groups"]
    for group_id in sorted(groups, key=lambda value: int(value)):
        names.append(f"land_cover__{groups[group_id]['name']}")
    return names


def create_figure(comparison: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    metrics = ["balanced_accuracy", "precision", "recall", "f2_score", "dice", "average_precision"]
    labels = ["Bal. accuracy", "Precision", "Recall", "F2", "Dice", "Avg. precision"]
    trivial = comparison[comparison["model"] == "trivial_prevalence"].iloc[0]
    logistic = comparison[comparison["model"] == "logistic_regression"].iloc[0]
    x = np.arange(len(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("ClimateAdapt AI – V3 Baseline Validation", fontsize=18, fontweight="bold")
    axes[0].bar(x - 0.2, [trivial[m] for m in metrics], 0.4, label="Trivial", color="#9ca3af")
    axes[0].bar(x + 0.2, [logistic[m] for m in metrics], 0.4, label="Logistic", color="#2563eb")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Validation metrics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].plot(thresholds["threshold"], thresholds["f2_score"], label="F2", color="#dc2626")
    axes[1].plot(thresholds["threshold"], thresholds["dice"], label="Dice", color="#7c3aed")
    best = thresholds.loc[thresholds["f2_score"].idxmax()]
    axes[1].axvline(best["threshold"], color="black", linestyle="--", label=f"Selected {best['threshold']:.3f}")
    axes[1].set_xlabel("Validation threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Logistic operational-threshold selection")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    for path in [MANIFEST_PATH, TILE_METADATA_PATH, REFERENCE_PATH]:
        if not path.exists():
            fail(f"Required input does not exist: {path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    tile_metadata = json.loads(TILE_METADATA_PATH.read_text(encoding="utf-8"))
    with rasterio.open(REFERENCE_PATH) as source:
        raster_width = source.width

    print("=" * 78)
    print("CLIMATEADAPT V3 – TRAIN AND VALIDATE BASELINE MODELS")
    print("=" * 78)
    print("Training split: Nelson CBD, Nelson Haven, Rocks Road, Tahunanui, Waimea inlet")
    print("Validation split: Boulder Bank Glenduan, Delaware Bay estuary")
    print("Locked test tile files will not be opened.")
    print("\nExtracting unique pixels from overlapping train and validation tiles...")
    x_train, y_train, train_unique, train_duplicates = load_unique_pixels(manifest, "train", raster_width)
    x_validation, y_validation, validation_unique, validation_duplicates = load_unique_pixels(manifest, "validation", raster_width)

    expected_channels = int(tile_metadata["input_channels"])
    if x_train.shape[1] != expected_channels or x_validation.shape[1] != expected_channels:
        fail("Tile channel count does not match V3 tile metadata.")

    print(f"Training:   {train_unique:,} unique pixels; duplicates merged={train_duplicates:,}; flood={int(y_train.sum()):,}")
    print(f"Validation: {validation_unique:,} unique pixels; duplicates merged={validation_duplicates:,}; flood={int(y_validation.sum()):,}")

    train_prevalence = float(y_train.mean())
    trivial_probabilities = np.full(len(y_validation), train_prevalence, dtype=np.float64)
    trivial_metrics = classification_metrics(y_validation, trivial_probabilities, 0.5)

    print("\nTraining logistic-regression baseline...")
    logistic_started = time.perf_counter()
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    logistic_seconds = time.perf_counter() - logistic_started
    validation_probabilities = model.predict_proba(x_validation)[:, 1]

    thresholds = threshold_search(y_validation, validation_probabilities)
    best_row = thresholds.sort_values(
        ["f2_score", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected_threshold = float(best_row["threshold"])
    logistic_metrics = classification_metrics(y_validation, validation_probabilities, selected_threshold)

    comparison = pd.DataFrame(
        [
            {"model": "trivial_prevalence", **trivial_metrics},
            {"model": "logistic_regression", **logistic_metrics},
        ]
    )
    comparison.to_csv(COMPARISON_PATH, index=False)
    thresholds.to_csv(THRESHOLD_PATH, index=False)

    names = feature_names(tile_metadata)
    coefficients = pd.DataFrame(
        {
            "feature": names,
            "coefficient": model.coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients["direction"] = np.where(
        coefficients["coefficient"] >= 0,
        "increases predicted susceptibility",
        "decreases predicted susceptibility",
    )
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(COEFFICIENT_PATH, index=False)

    model_artifact = {
        "model": model,
        "feature_names": names,
        "selected_validation_f2_threshold": selected_threshold,
        "random_seed": RANDOM_SEED,
        "training_prevalence": train_prevalence,
        "tile_metadata_path": str(TILE_METADATA_PATH.relative_to(PROJECT_ROOT)),
        "test_accessed": False,
    }
    joblib.dump(model_artifact, MODEL_PATH)

    selected_payload = {
        "model": "logistic_regression_v3",
        "selection_split": "validation",
        "selection_objective": "maximum F2 score",
        "selected_threshold": selected_threshold,
        "validation_metrics": logistic_metrics,
        "test_accessed": False,
    }
    SELECTED_THRESHOLD_PATH.write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    summary = {
        "experiment": "ClimateAdapt V3 baselines",
        "test_accessed": False,
        "training_unique_pixels": train_unique,
        "validation_unique_pixels": validation_unique,
        "training_overlap_duplicates_merged": train_duplicates,
        "validation_overlap_duplicates_merged": validation_duplicates,
        "training_prevalence": train_prevalence,
        "validation_prevalence": float(y_validation.mean()),
        "input_channels": expected_channels,
        "logistic_training_seconds": logistic_seconds,
        "trivial_validation_metrics": trivial_metrics,
        "logistic_validation_metrics": logistic_metrics,
        "logistic_beats_trivial_average_precision": (
            logistic_metrics["average_precision"] > trivial_metrics["average_precision"]
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_figure(comparison, thresholds)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("V3 BASELINE VALIDATION COMPLETED")
    print("=" * 78)
    print("TRIVIAL TRAINING-PREVALENCE BASELINE")
    print(f"Validation AP:           {trivial_metrics['average_precision']:.4f}")
    print(f"Validation balanced acc: {trivial_metrics['balanced_accuracy']:.4f}")
    print(f"Validation F2:           {trivial_metrics['f2_score']:.4f}")
    print("-" * 78)
    print("LOGISTIC REGRESSION BASELINE")
    print(f"Selected threshold:      {selected_threshold:.3f}")
    print(f"Validation AP:           {logistic_metrics['average_precision']:.4f}")
    print(f"Validation ROC-AUC:      {logistic_metrics['roc_auc']:.4f}")
    print(f"Validation precision:    {logistic_metrics['precision']:.4f}")
    print(f"Validation recall:       {logistic_metrics['recall']:.4f}")
    print(f"Validation F2:           {logistic_metrics['f2_score']:.4f}")
    print(f"Validation Dice:         {logistic_metrics['dice']:.4f}")
    print(f"Training time:           {logistic_seconds:.2f} seconds")
    print("-" * 78)
    print(f"Model:        {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary:      {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Comparison:   {COMPARISON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Thresholds:   {THRESHOLD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Coefficients: {COEFFICIENT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure:       {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Total time:   {elapsed:.2f} seconds")
    print("=" * 78)
    print("The locked test tile folder was not opened.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBaseline training cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
