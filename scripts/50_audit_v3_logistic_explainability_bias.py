"""Audit the deployed ClimateAdapt V3 logistic model without changing it.

This post-hoc audit opens the already-unlocked V3 geographic test tiles,
deduplicates overlapping pixels, and reports:

1. exact linear-model coefficient and logit-contribution attribution; and
2. performance across geographic and environmental subgroups.

It does not fit a model, search a threshold, alter a feature, or write into any
training/validation artifact. The audit is environmental/spatial and must not
be described as demographic fairness analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TILE_ROOT = PROJECT_ROOT / "data" / "processed" / "v3" / "tiles"
MANIFEST_PATH = TILE_ROOT / "v3_tile_manifest.csv"
TILE_METADATA_PATH = TILE_ROOT / "v3_tile_metadata.json"
REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "v3" / "v3_split_mask_5m.tif"
MODEL_PATH = PROJECT_ROOT / "models" / "v3" / "logistic_regression_v3.joblib"
THRESHOLD_PATH = (
    PROJECT_ROOT / "outputs" / "baselines_v3" / "logistic_selected_threshold_v3.json"
)
LOCKED_TEST_SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "test_v3" / "v3_locked_test_summary.json"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "explainability_bias_v3"
FIGURE_ROOT = PROJECT_ROOT / "evidence" / "figures" / "v3"
ATTRIBUTION_CSV = OUTPUT_ROOT / "v3_logistic_exact_attribution.csv"
SUBGROUP_CSV = OUTPUT_ROOT / "v3_environmental_subgroup_metrics.csv"
RANGE_CSV = OUTPUT_ROOT / "v3_subgroup_performance_ranges.csv"
SUMMARY_JSON = OUTPUT_ROOT / "v3_explainability_bias_audit.json"
FIGURE_PATH = FIGURE_ROOT / "v3_logistic_explainability_bias_audit.png"

AREA_NAMES = {3: "Delaware Open Coast", 6: "Oananga Bay"}
EXPECTED_TEST_AREAS = set(AREA_NAMES)
RANDOM_SEED = 8532
MIN_SUBGROUP_PIXELS = 1_000
MIN_SUBGROUP_POSITIVES = 100
RECALL_RANGE_WARNING = 0.20
PRECISION_RANGE_WARNING = 0.20


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def unwrap_estimator(artifact):
    if hasattr(artifact, "predict_proba"):
        return artifact, {}
    if isinstance(artifact, dict):
        for key in ("model", "pipeline", "estimator", "classifier"):
            candidate = artifact.get(key)
            if hasattr(candidate, "predict_proba"):
                return candidate, artifact
    raise TypeError("The V3 logistic artifact has no predict_proba estimator.")


def feature_names(metadata: dict) -> list[str]:
    names = ["elevation", "slope", "log1p_distance_to_coast"]
    groups = metadata["land_cover_semantic_groups"]
    for group_id in sorted(groups, key=lambda value: int(value)):
        names.append(f"land_cover__{groups[group_id]['name']}")
    return names


def resolve_tile_path(relative_path: str) -> Path:
    normalised = str(relative_path).replace("\\", "/")
    for candidate in (TILE_ROOT / normalised, PROJECT_ROOT / normalised, Path(normalised)):
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
        image = array_from_any(tile, ("features", "image", "x"), path).astype(np.float32)
        target = np.squeeze(
            array_from_any(tile, ("target", "y", "mask"), path).astype(np.uint8)
        )
        valid = np.squeeze(
            array_from_any(tile, ("valid_mask", "valid"), path).astype(bool)
        )
        if "area_zone_id" in tile.files:
            area = np.squeeze(np.asarray(tile["area_zone_id"], dtype=np.uint8))
        elif "area_id" in tile.files:
            area = np.squeeze(np.asarray(tile["area_id"], dtype=np.uint8))
        else:
            area = np.zeros(target.shape, dtype=np.uint8)

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
    shape = channels_first.shape[1:]
    if target.shape != shape or valid.shape != shape or area.shape != shape:
        raise ValueError(
            f"Spatial mismatch in {path.name}: features={shape}, target={target.shape}, "
            f"valid={valid.shape}, area={area.shape}"
        )
    return channels_first, target, valid, area


def load_unique_test_pixels(
    manifest: pd.DataFrame,
    raster_width: int,
    expected_channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    records = manifest[
        manifest["split"].astype(str).str.lower().str.strip().isin(("test", "testing", "3"))
    ]
    if records.empty:
        fail("No V3 test records were found in the tile manifest.")

    index_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    area_parts: list[np.ndarray] = []
    for sequence, (_, row) in enumerate(records.iterrows(), start=1):
        path = resolve_tile_path(str(row["file"]))
        image, target, valid, area = load_tile(path, expected_channels)
        if not np.any(area) and "area_id" in row.index:
            area.fill(int(row["area_id"]))
        elif not np.any(area) and "area_zone_id" in row.index:
            area.fill(int(row["area_zone_id"]))

        local_rows, local_columns = np.nonzero(valid)
        row_start = int(row["row"])
        column_start = int(row["column"])
        global_indices = (
            (row_start + local_rows).astype(np.int64) * raster_width
            + (column_start + local_columns).astype(np.int64)
        )
        index_parts.append(global_indices)
        feature_parts.append(image[:, valid].T.astype(np.float32, copy=False))
        target_parts.append(target[valid].astype(np.uint8, copy=False))
        area_parts.append(area[valid].astype(np.uint8, copy=False))
        if sequence % 50 == 0 or sequence == len(records):
            print(f"Loaded test tile {sequence}/{len(records)}")

    indices = np.concatenate(index_parts)
    features = np.concatenate(feature_parts)
    truth = np.concatenate(target_parts)
    areas = np.concatenate(area_parts)
    _, first_positions = np.unique(indices, return_index=True)
    first_positions.sort()
    duplicates = int(indices.size - first_positions.size)
    return (
        features[first_positions],
        truth[first_positions],
        areas[first_positions],
        duplicates,
    )


def inverse_numeric_features(features: np.ndarray, metadata: dict) -> dict[str, np.ndarray]:
    normalisation = metadata["normalisation"]
    elevation = (
        features[:, 0] * float(normalisation["elevation"]["std"])
        + float(normalisation["elevation"]["mean"])
    )
    slope = (
        features[:, 1] * float(normalisation["slope"]["std"])
        + float(normalisation["slope"]["mean"])
    )
    log_distance = (
        features[:, 2] * float(normalisation["log1p_distance_to_coast"]["std"])
        + float(normalisation["log1p_distance_to_coast"]["mean"])
    )
    return {
        "elevation_metres": elevation.astype(np.float32),
        "slope_degrees": slope.astype(np.float32),
        "distance_to_coast_metres": np.expm1(log_distance).astype(np.float32),
    }


def safe_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(truth).size < 2:
        return float("nan")
    return float(roc_auc_score(truth, scores))


def safe_ap(truth: np.ndarray, scores: np.ndarray) -> float:
    if np.sum(truth) == 0:
        return float("nan")
    return float(average_precision_score(truth, scores))


def subgroup_record(
    group_type: str,
    group_id: str,
    group_label: str,
    selector: np.ndarray,
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    y = truth[selector]
    probability = scores[selector]
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    positives = int(np.sum(y))
    eligible = int(y.size) >= MIN_SUBGROUP_PIXELS and positives >= MIN_SUBGROUP_POSITIVES
    return {
        "group_type": group_type,
        "group_id": group_id,
        "group_label": group_label,
        "pixels": int(y.size),
        "positive_pixels": positives,
        "prevalence": float(np.mean(y)) if y.size else float("nan"),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, predicted))
            if np.unique(y).size == 2
            else float("nan")
        ),
        "f2_score": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "average_precision": safe_ap(y, probability),
        "roc_auc": safe_auc(y, probability),
        "eligible_for_range_comparison": eligible,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def add_binned_records(
    records: list[dict],
    group_type: str,
    values: np.ndarray,
    edges: list[float],
    labels: list[str],
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    bins = np.digitize(values, edges[1:-1], right=False)
    for index, label in enumerate(labels):
        selector = bins == index
        if np.any(selector):
            records.append(
                subgroup_record(
                    group_type, str(index), label, selector, truth, scores, threshold
                )
            )


def performance_ranges(subgroups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("recall", "precision", "false_negative_rate", "average_precision")
    for group_type, frame in subgroups.groupby("group_type", sort=False):
        eligible = frame[frame["eligible_for_range_comparison"]].copy()
        for metric in metrics:
            values = pd.to_numeric(eligible[metric], errors="coerce").dropna()
            if values.empty:
                minimum = maximum = value_range = float("nan")
            else:
                minimum = float(values.min())
                maximum = float(values.max())
                value_range = maximum - minimum
            rows.append(
                {
                    "group_type": group_type,
                    "metric": metric,
                    "eligible_groups": int(values.size),
                    "minimum": minimum,
                    "maximum": maximum,
                    "range": value_range,
                }
            )
    return pd.DataFrame(rows)


def create_figure(
    attribution: pd.DataFrame,
    subgroups: pd.DataFrame,
    ranges: pd.DataFrame,
    threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    fig.suptitle(
        "ClimateAdapt AI – V3 Logistic Explainability and Environmental Bias Audit",
        fontsize=18,
        fontweight="bold",
    )

    ordered = attribution.sort_values("absolute_coefficient", ascending=True)
    coefficient_colours = np.where(
        ordered["coefficient"] >= 0, "#dc2626", "#2563eb"
    )
    axes[0, 0].barh(
        ordered["display_name"],
        ordered["coefficient"],
        color=coefficient_colours,
    )
    axes[0, 0].axvline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Exact signed logistic coefficients")
    axes[0, 0].set_xlabel("Change in log-odds per encoded feature unit")
    axes[0, 0].grid(axis="x", alpha=0.25)

    contribution = attribution.sort_values("mean_absolute_logit_contribution", ascending=True)
    axes[0, 1].barh(
        contribution["display_name"],
        contribution["mean_absolute_logit_contribution"],
        color="#0f766e",
    )
    axes[0, 1].set_title("Mean absolute contribution on locked test pixels")
    axes[0, 1].set_xlabel("Mean absolute contribution to model logit")
    axes[0, 1].grid(axis="x", alpha=0.25)

    geography = subgroups[subgroups["group_type"] == "geography"]
    x = np.arange(len(geography))
    axes[1, 0].bar(x - 0.18, geography["precision"], 0.36, label="Precision", color="#f59e0b")
    axes[1, 0].bar(x + 0.18, geography["recall"], 0.36, label="Recall", color="#16a34a")
    axes[1, 0].set_xticks(x, geography["group_label"], rotation=12, ha="right")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title(f"Geographic performance at frozen threshold {threshold:.3f}")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend()

    recall_ranges = ranges[ranges["metric"] == "recall"]
    axes[1, 1].bar(
        recall_ranges["group_type"].str.replace("_", " "),
        recall_ranges["range"],
        color="#7c3aed",
    )
    axes[1, 1].axhline(
        RECALL_RANGE_WARNING,
        color="#dc2626",
        linestyle="--",
        label=f"Review trigger {RECALL_RANGE_WARNING:.2f}",
    )
    axes[1, 1].set_ylim(0, max(1.0, float(recall_ranges["range"].max()) * 1.15))
    axes[1, 1].set_ylabel("Recall range (maximum − minimum)")
    axes[1, 1].set_title("Environmental subgroup disparity screening")
    axes[1, 1].tick_params(axis="x", rotation=22)
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend()

    fig.text(
        0.5,
        0.005,
        "Attribution describes this fitted model; it does not establish causation. "
        "Subgroups are environmental/spatial, not demographic groups.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    required = [
        MANIFEST_PATH,
        TILE_METADATA_PATH,
        REFERENCE_PATH,
        MODEL_PATH,
        THRESHOLD_PATH,
        LOCKED_TEST_SUMMARY_PATH,
    ]
    for path in required:
        if not path.exists():
            fail(f"Required input does not exist: {path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    metadata = read_json(TILE_METADATA_PATH)
    threshold_information = read_json(THRESHOLD_PATH)
    locked_test_summary = read_json(LOCKED_TEST_SUMMARY_PATH)
    threshold = float(threshold_information["selected_threshold"])
    if not math.isclose(threshold, 0.004, rel_tol=0.0, abs_tol=1e-12):
        fail(f"Expected frozen V3 logistic threshold 0.004; found {threshold}.")

    artifact = joblib.load(MODEL_PATH)
    estimator, artifact_metadata = unwrap_estimator(artifact)
    names = artifact_metadata.get("feature_names") or feature_names(metadata)
    expected_channels = int(metadata["input_channels"])
    if len(names) != expected_channels:
        fail("Feature-name count does not match the V3 channel count.")
    if int(getattr(estimator, "n_features_in_", expected_channels)) != expected_channels:
        fail("Saved logistic model feature count does not match V3 tile metadata.")

    manifest = pd.read_csv(MANIFEST_PATH)
    required_manifest_columns = {"split", "file", "row", "column"}
    if not required_manifest_columns.issubset(manifest.columns):
        fail(
            "V3 tile manifest is missing required columns: "
            + ", ".join(sorted(required_manifest_columns - set(manifest.columns)))
        )
    with rasterio.open(REFERENCE_PATH) as source:
        raster_width = source.width

    print("=" * 78)
    print("CLIMATEADAPT V3 – LOGISTIC EXPLAINABILITY AND DATA-BIAS AUDIT")
    print("=" * 78)
    print("Model: deployed V3 logistic regression")
    print("Method: exact linear attribution plus environmental subgroup audit")
    print("Test regions: Delaware Open Coast + Oananga Bay")
    print(f"Frozen operational threshold: {threshold:.3f}")
    print("No model weights, features, splits or thresholds will be changed.")
    print("Loading and deduplicating already-unlocked V3 test pixels...")

    features, truth, areas, duplicates = load_unique_test_pixels(
        manifest, raster_width, expected_channels
    )
    observed_areas = set(np.unique(areas).tolist())
    if observed_areas != EXPECTED_TEST_AREAS:
        fail(
            f"Expected test area IDs {sorted(EXPECTED_TEST_AREAS)}, observed "
            f"{sorted(observed_areas)}."
        )
    scores = estimator.predict_proba(features)[:, 1]
    coefficients = np.asarray(estimator.coef_).reshape(-1)
    contributions = features * coefficients[None, :]

    attribution = pd.DataFrame(
        {
            "feature": names,
            "display_name": [name.replace("land_cover__", "LC: ").replace("_", " ") for name in names],
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "mean_feature_value": np.mean(features, axis=0),
            "mean_logit_contribution": np.mean(contributions, axis=0),
            "mean_absolute_logit_contribution": np.mean(np.abs(contributions), axis=0),
            "direction": np.where(
                coefficients >= 0,
                "increases fitted susceptibility score",
                "decreases fitted susceptibility score",
            ),
        }
    ).sort_values("mean_absolute_logit_contribution", ascending=False)
    attribution.to_csv(ATTRIBUTION_CSV, index=False)

    physical = inverse_numeric_features(features, metadata)
    subgroup_records: list[dict] = []
    for area_id, area_name in AREA_NAMES.items():
        subgroup_records.append(
            subgroup_record(
                "geography",
                str(area_id),
                area_name,
                areas == area_id,
                truth,
                scores,
                threshold,
            )
        )

    add_binned_records(
        subgroup_records,
        "elevation_band",
        physical["elevation_metres"],
        [-np.inf, 2.0, 5.0, 10.0, np.inf],
        ["≤2 m", "2–5 m", "5–10 m", ">10 m"],
        truth,
        scores,
        threshold,
    )
    add_binned_records(
        subgroup_records,
        "slope_band",
        physical["slope_degrees"],
        [-np.inf, 2.0, 5.0, 15.0, np.inf],
        ["<2°", "2–5°", "5–15°", ">15°"],
        truth,
        scores,
        threshold,
    )
    add_binned_records(
        subgroup_records,
        "coast_distance_band",
        physical["distance_to_coast_metres"],
        [-np.inf, 100.0, 500.0, 1000.0, np.inf],
        ["<100 m", "100–500 m", "500–1,000 m", ">1,000 m"],
        truth,
        scores,
        threshold,
    )

    land_cover_groups = np.argmax(features[:, 3:], axis=1)
    definitions = metadata["land_cover_semantic_groups"]
    for group_id in sorted(definitions, key=lambda value: int(value)):
        numeric_id = int(group_id)
        selector = land_cover_groups == numeric_id
        if np.any(selector):
            subgroup_records.append(
                subgroup_record(
                    "land_cover_group",
                    group_id,
                    definitions[group_id]["name"].replace("_", " "),
                    selector,
                    truth,
                    scores,
                    threshold,
                )
            )

    subgroups = pd.DataFrame(subgroup_records)
    subgroups.to_csv(SUBGROUP_CSV, index=False)
    ranges = performance_ranges(subgroups)
    ranges.to_csv(RANGE_CSV, index=False)

    warnings: list[str] = []
    for _, row in ranges.iterrows():
        if not np.isfinite(row["range"]):
            continue
        if row["metric"] == "recall" and row["range"] > RECALL_RANGE_WARNING:
            warnings.append(
                f"{row['group_type']} recall range is {row['range']:.4f}, above the "
                f"predeclared {RECALL_RANGE_WARNING:.2f} review trigger."
            )
        if row["metric"] == "precision" and row["range"] > PRECISION_RANGE_WARNING:
            warnings.append(
                f"{row['group_type']} precision range is {row['range']:.4f}, above the "
                f"predeclared {PRECISION_RANGE_WARNING:.2f} review trigger."
            )
    excluded = subgroups[~subgroups["eligible_for_range_comparison"]]
    if not excluded.empty:
        warnings.append(
            f"{len(excluded)} subgroup(s) were descriptive only because they had fewer "
            f"than {MIN_SUBGROUP_PIXELS:,} pixels or {MIN_SUBGROUP_POSITIVES:,} positives."
        )

    strongest = attribution.iloc[0]
    pooled_prediction = scores >= threshold
    pooled_recall = float(recall_score(truth, pooled_prediction, zero_division=0))
    pooled_precision = float(precision_score(truth, pooled_prediction, zero_division=0))
    runtime = time.perf_counter() - started
    create_figure(attribution, subgroups, ranges, threshold)

    summary = {
        "experiment": "ClimateAdapt V3 deployed logistic explainability and data-bias audit",
        "audit_type": "post-hoc; no tuning or model selection",
        "model": "V3 logistic regression",
        "model_tribe": "symbolist/statistical linear classifier",
        "model_sha256": sha256(MODEL_PATH),
        "operational_threshold": threshold,
        "model_intercept": float(np.asarray(estimator.intercept_).reshape(-1)[0]),
        "threshold_source": "V3 validation split (Boulder Bank and Delaware Bay)",
        "test_data_status": "already unlocked by Script 44; reused for audit only",
        "test_regions": list(AREA_NAMES.values()),
        "unique_test_pixels": int(truth.size),
        "overlapping_duplicate_pixels_merged": duplicates,
        "positive_test_pixels": int(np.sum(truth)),
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "pooled_average_precision": float(average_precision_score(truth, scores)),
        "strongest_mean_absolute_logit_contribution": {
            "feature": strongest["feature"],
            "value": float(strongest["mean_absolute_logit_contribution"]),
            "coefficient": float(strongest["coefficient"]),
        },
        "explainability_method": {
            "name": "exact linear logit decomposition",
            "equivalent_role": "transparent-model attribution equivalent for the fitted logistic classifier",
            "calculation": "model intercept plus the sum of feature value multiplied by fitted coefficient",
            "proves": "how encoded features contribute to this fitted model's logit",
            "does_not_prove": [
                "causation",
                "physical flood mechanism",
                "feature independence",
                "future generalisation",
                "absence of bias",
            ],
        },
        "subgroup_audit": {
            "scope": "environmental and spatial data-bias audit; not demographic fairness",
            "group_types": list(subgroups["group_type"].drop_duplicates()),
            "minimum_pixels_for_range": MIN_SUBGROUP_PIXELS,
            "minimum_positive_pixels_for_range": MIN_SUBGROUP_POSITIVES,
            "recall_range_review_trigger": RECALL_RANGE_WARNING,
            "precision_range_review_trigger": PRECISION_RANGE_WARNING,
            "ranges": ranges.to_dict(orient="records"),
        },
        "warnings": warnings,
        "locked_test_summary_sha256": sha256(LOCKED_TEST_SUMMARY_PATH),
        "no_model_weights_changed": True,
        "no_threshold_changed": True,
        "no_split_changed": True,
        "no_feature_engineering_changed": True,
        "no_retraining_performed": True,
        "test_results_used_for_tuning": False,
        "processing_time_seconds": runtime,
        "artifacts": {
            "attribution_csv": str(ATTRIBUTION_CSV.relative_to(PROJECT_ROOT)),
            "subgroup_csv": str(SUBGROUP_CSV.relative_to(PROJECT_ROOT)),
            "range_csv": str(RANGE_CSV.relative_to(PROJECT_ROOT)),
            "figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    SUMMARY_JSON.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    max_recall_range = pd.to_numeric(
        ranges.loc[ranges["metric"] == "recall", "range"], errors="coerce"
    ).max()
    print("\n" + "=" * 78)
    print("V3 LOGISTIC EXPLAINABILITY AND DATA-BIAS AUDIT COMPLETED")
    print("=" * 78)
    print(f"Unique test pixels:          {truth.size:,}")
    print(f"Pooled average precision:   {average_precision_score(truth, scores):.4f}")
    print(f"Pooled precision / recall:  {pooled_precision:.4f} / {pooled_recall:.4f}")
    print(f"Strongest test contribution:{strongest['feature']}")
    print(f"Maximum subgroup recall gap:{max_recall_range:.4f}")
    print(f"Warnings recorded:          {len(warnings)}")
    print("-" * 78)
    print(f"Attribution CSV: {ATTRIBUTION_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Subgroup CSV:    {SUBGROUP_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Range CSV:       {RANGE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Summary JSON:    {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Evidence figure: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Processing time: {runtime:.2f} seconds")
    print("=" * 78)
    print("No model weights, features, splits or thresholds were changed.")
    print("Attribution describes the fitted model and does not prove causation.")
    print("This is environmental/spatial data-bias auditing, not demographic fairness.")


if __name__ == "__main__":
    main()
