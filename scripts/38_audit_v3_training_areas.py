"""Audit full-region data availability before ClimateAdapt V3 training.

This script is intentionally read-only with respect to V1/V2 artifacts. It
uses the prepared full-region 5 m rasters to measure each NCC coastal area,
its surrounding valid-land context, feature distributions, land-cover
representation and candidate tile availability.

It does not create a final split, train a model, select a threshold or read
any V1/V2 prediction raster.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RASTER_ROOT = PROJECT_ROOT / "data" / "processed" / "full_region" / "rasters"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v3_audit"
FIGURE_ROOT = PROJECT_ROOT / "evidence" / "figures" / "v3"

METADATA_PATH = RASTER_ROOT / "full_region_metadata.json"
RASTER_PATHS = {
    "elevation": RASTER_ROOT / "elevation_5m.tif",
    "slope": RASTER_ROOT / "slope_degrees_5m.tif",
    "distance": RASTER_ROOT / "distance_to_coast_5m.tif",
    "land_cover": RASTER_ROOT / "land_cover_class_5m.tif",
    "valid": RASTER_ROOT / "valid_land_mask_5m.tif",
    "flood_target": RASTER_ROOT / "flood_target_5m.tif",
    "area_id": RASTER_ROOT / "flood_area_id_5m.tif",
}

CONTEXT_METRES = 750.0
TILE_SIZE = 128
STRIDE = 64
MIN_CONTEXT_FRACTION = 0.20

# These are hypotheses to audit, not a frozen split.
PREFERRED_ROLE = {
    "Rocks Road": "validation_candidate",
    "Delaware Open Coast": "locked_test_candidate",
    "Oananga Bay": "locked_test_candidate",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def verify_inputs() -> dict:
    missing = [str(path) for path in [METADATA_PATH, *RASTER_PATHS.values()] if not path.exists()]
    if missing:
        fail("Missing required files:\n  " + "\n  ".join(missing))

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    reference = None
    for label, path in RASTER_PATHS.items():
        with rasterio.open(path) as source:
            signature = (
                source.width,
                source.height,
                str(source.crs),
                tuple(source.transform),
            )
            if reference is None:
                reference = signature
            elif signature != reference:
                fail(f"Raster is not aligned with the reference grid: {label} ({path})")

    return metadata


def invert_mapping(metadata: dict) -> dict[int, str]:
    raw_mapping = metadata.get("flood_area_id_mapping")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        fail("full_region_metadata.json has no flood_area_id_mapping object.")
    mapping = {int(area_id): str(name) for name, area_id in raw_mapping.items()}
    if len(mapping) != len(raw_mapping):
        fail("Flood-area IDs are not unique.")
    return dict(sorted(mapping.items()))


def expanded_window(rows: np.ndarray, cols: np.ndarray, padding: int, height: int, width: int) -> Window:
    row_start = max(0, int(rows.min()) - padding)
    row_stop = min(height, int(rows.max()) + padding + 1)
    col_start = max(0, int(cols.min()) - padding)
    col_stop = min(width, int(cols.max()) + padding + 1)
    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def finite_values(array: np.ndarray, mask: np.ndarray, nodata) -> np.ndarray:
    keep = mask & np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        keep &= array != nodata
    return array[keep].astype(np.float64, copy=False)


def stats(prefix: str, values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p05": None,
            f"{prefix}_p95": None,
        }
    p05, median, p95 = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(median),
        f"{prefix}_p05": float(p05),
        f"{prefix}_p95": float(p95),
    }


def aligned_starts(first: int, last: int, stride: int, limit: int, tile_size: int) -> list[int]:
    start = max(0, (first // stride) * stride)
    final = min(limit - tile_size, int(math.ceil(last / stride)) * stride)
    if final < start:
        return []
    return list(range(start, final + 1, stride))


def tile_availability(
    context_mask: np.ndarray,
    current_area_mask: np.ndarray,
    all_area_ids: np.ndarray,
    window: Window,
    full_height: int,
    full_width: int,
) -> dict:
    row0 = int(window.row_off)
    col0 = int(window.col_off)
    row1 = row0 + int(window.height) - 1
    col1 = col0 + int(window.width) - 1
    row_starts = aligned_starts(row0, row1, STRIDE, full_height, TILE_SIZE)
    col_starts = aligned_starts(col0, col1, STRIDE, full_width, TILE_SIZE)

    minimum_context = int(TILE_SIZE * TILE_SIZE * MIN_CONTEXT_FRACTION)
    positive = 0
    negative = 0
    excluded_other_area = 0
    too_little_context = 0

    for global_row in row_starts:
        local_row = global_row - row0
        if local_row < 0 or local_row + TILE_SIZE > context_mask.shape[0]:
            continue
        for global_col in col_starts:
            local_col = global_col - col0
            if local_col < 0 or local_col + TILE_SIZE > context_mask.shape[1]:
                continue
            rs = slice(local_row, local_row + TILE_SIZE)
            cs = slice(local_col, local_col + TILE_SIZE)
            tile_context = context_mask[rs, cs]
            if int(tile_context.sum()) < minimum_context:
                too_little_context += 1
                continue

            tile_current = current_area_mask[rs, cs] & tile_context
            if np.any(tile_current):
                positive += 1
                continue

            tile_other = (all_area_ids[rs, cs] > 0) & tile_context
            if np.any(tile_other):
                excluded_other_area += 1
            else:
                negative += 1

    return {
        "candidate_positive_tiles": positive,
        "candidate_negative_tiles": negative,
        "candidate_tiles_total": positive + negative,
        "tiles_excluded_other_flood_area": excluded_other_area,
        "tiles_rejected_low_context": too_little_context,
    }


def availability_rating(positive_tiles: int, negative_tiles: int) -> tuple[str, list[str]]:
    warnings = []
    if positive_tiles < 5:
        warnings.append("fewer than 5 positive tiles")
    elif positive_tiles < 15:
        warnings.append("fewer than 15 positive tiles")
    if negative_tiles < 5:
        warnings.append("fewer than 5 negative tiles")
    elif negative_tiles < 15:
        warnings.append("fewer than 15 negative tiles")

    if positive_tiles >= 20 and negative_tiles >= 20:
        rating = "strong"
    elif positive_tiles >= 10 and negative_tiles >= 10:
        rating = "usable"
    elif positive_tiles >= 5 and negative_tiles >= 5:
        rating = "limited"
    else:
        rating = "insufficient"
    return rating, warnings


def build_recommendation(records: list[dict]) -> dict:
    areas = []
    blockers = []
    for record in records:
        name = record["coastal_area"]
        preferred = PREFERRED_ROLE.get(name, "development_candidate")
        suitable = record["availability_rating"] in {"strong", "usable"}
        if preferred == "locked_test_candidate" and not suitable:
            blockers.append(
                f"{name} cannot yet be locked for testing because its tile availability is "
                f"{record['availability_rating']}."
            )
        areas.append(
            {
                "coastal_area": name,
                "area_id": record["area_id"],
                "preferred_role_before_audit": preferred,
                "availability_rating": record["availability_rating"],
                "candidate_positive_tiles": record["candidate_positive_tiles"],
                "candidate_negative_tiles": record["candidate_negative_tiles"],
                "eligible_for_preferred_role": suitable,
                "warnings": record["warnings"],
            }
        )

    return {
        "status": "ready_for_split_design" if not blockers else "split_design_requires_revision",
        "important": "This is a data-based recommendation, not the frozen V3 split.",
        "context_metres": CONTEXT_METRES,
        "tile_size_pixels": TILE_SIZE,
        "stride_pixels": STRIDE,
        "minimum_context_fraction": MIN_CONTEXT_FRACTION,
        "areas": areas,
        "blocking_findings": blockers,
        "next_action": "Review this audit and explicitly freeze the V3 geographic split before tile creation.",
    }


def create_figure(frame: pd.DataFrame, output_path: Path) -> None:
    order = frame.sort_values("area_id").copy()
    labels = [name.replace(" ", "\n") for name in order["coastal_area"]]
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    fig.suptitle("ClimateAdapt AI – V3 Full-Region Area Audit", fontsize=20, fontweight="bold")

    axes[0, 0].bar(x, order["official_flood_area_km2"], color="#b91c1c")
    axes[0, 0].set_title("Official flood-labelled valid land")
    axes[0, 0].set_ylabel("Area (km²)")

    axes[0, 1].bar(x, order["context_valid_land_area_km2"], color="#2563eb")
    axes[0, 1].set_title(f"Valid-land context within {CONTEXT_METRES:.0f} m")
    axes[0, 1].set_ylabel("Area (km²)")

    width = 0.38
    axes[1, 0].bar(
        x - width / 2,
        order["candidate_positive_tiles"],
        width,
        label="Contains area flood pixels",
        color="#dc2626",
    )
    axes[1, 0].bar(
        x + width / 2,
        order["candidate_negative_tiles"],
        width,
        label="No flood pixels",
        color="#4f83b3",
    )
    axes[1, 0].set_title("Candidate 128×128 tiles")
    axes[1, 0].set_ylabel("Tiles")
    axes[1, 0].legend()

    axes[1, 1].bar(x, order["context_flood_percentage"], color="#7c3aed")
    axes[1, 1].set_title("Flood prevalence in local context")
    axes[1, 1].set_ylabel("Flood-labelled valid land (%)")

    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)

    fig.text(
        0.5,
        0.003,
        "Audit evidence only: no V3 split, threshold or model has been selected.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    metadata = verify_inputs()
    area_mapping = invert_mapping(metadata)

    print("=" * 78)
    print("CLIMATEADAPT AI – V3 FULL-REGION TRAINING-AREA AUDIT")
    print("=" * 78)
    print("Purpose: audit data availability before any V3 split or training")
    print(f"Context distance: {CONTEXT_METRES:.0f} metres")
    print(f"Tile size / stride: {TILE_SIZE} / {STRIDE} pixels")
    print("V1/V2 model outputs will not be accessed or modified.")

    with rasterio.open(RASTER_PATHS["area_id"]) as source:
        area_ids_full = source.read(1)
        full_height, full_width = source.height, source.width
        resolution = abs(float(source.transform.a))

    with rasterio.open(RASTER_PATHS["valid"]) as source:
        valid_full = source.read(1) == 1

    cell_area_km2 = (resolution * resolution) / 1_000_000.0
    padding_pixels = int(math.ceil(CONTEXT_METRES / resolution))

    records: list[dict] = []
    land_cover_records: list[dict] = []

    datasets = {name: rasterio.open(path) for name, path in RASTER_PATHS.items() if name not in {"area_id", "valid"}}
    try:
        for area_id, area_name in area_mapping.items():
            print(f"\nAuditing {area_id}: {area_name}")
            valid_positive_full = (area_ids_full == area_id) & valid_full
            rows, cols = np.nonzero(valid_positive_full)
            if rows.size == 0:
                print("  WARNING: no labelled flood cells on valid land")
                records.append(
                    {
                        "area_id": area_id,
                        "coastal_area": area_name,
                        "official_flood_cells": 0,
                        "official_flood_area_km2": 0.0,
                        "context_valid_land_cells": 0,
                        "context_valid_land_area_km2": 0.0,
                        "context_flood_percentage": 0.0,
                        "candidate_positive_tiles": 0,
                        "candidate_negative_tiles": 0,
                        "candidate_tiles_total": 0,
                        "tiles_excluded_other_flood_area": 0,
                        "tiles_rejected_low_context": 0,
                        "availability_rating": "insufficient",
                        "warnings": ["no labelled flood cells on valid land"],
                    }
                )
                continue

            window = expanded_window(rows, cols, padding_pixels, full_height, full_width)
            row0, col0 = int(window.row_off), int(window.col_off)
            height, width = int(window.height), int(window.width)
            local_slice = (
                slice(row0, row0 + height),
                slice(col0, col0 + width),
            )
            local_ids = area_ids_full[local_slice]
            local_valid = valid_full[local_slice]
            current_area = local_ids == area_id

            distance_pixels = distance_transform_edt(~current_area)
            context_mask = local_valid & (distance_pixels * resolution <= CONTEXT_METRES)
            flood_mask = current_area & local_valid
            all_flood_mask = (local_ids > 0) & local_valid

            official_cells = int(flood_mask.sum())
            context_cells = int(context_mask.sum())
            context_flood_cells = int((all_flood_mask & context_mask).sum())

            record = {
                "area_id": area_id,
                "coastal_area": area_name,
                "official_flood_cells": official_cells,
                "official_flood_area_km2": official_cells * cell_area_km2,
                "context_valid_land_cells": context_cells,
                "context_valid_land_area_km2": context_cells * cell_area_km2,
                "context_flood_cells_all_areas": context_flood_cells,
                "context_flood_percentage": (100.0 * context_flood_cells / context_cells) if context_cells else 0.0,
                "bbox_row_min": int(rows.min()),
                "bbox_row_max": int(rows.max()),
                "bbox_column_min": int(cols.min()),
                "bbox_column_max": int(cols.max()),
            }

            for feature_name in ["elevation", "slope", "distance"]:
                dataset = datasets[feature_name]
                array = dataset.read(1, window=window)
                record.update(stats(f"{feature_name}_context", finite_values(array, context_mask, dataset.nodata)))
                record.update(stats(f"{feature_name}_flood", finite_values(array, flood_mask, dataset.nodata)))

            land_cover = datasets["land_cover"].read(1, window=window)
            for scope, mask in [("context", context_mask), ("flood", flood_mask)]:
                counts = Counter(int(value) for value in land_cover[mask])
                total = sum(counts.values())
                for class_code, count in sorted(counts.items()):
                    land_cover_records.append(
                        {
                            "area_id": area_id,
                            "coastal_area": area_name,
                            "scope": scope,
                            "class_code": class_code,
                            "cells": count,
                            "percentage": (100.0 * count / total) if total else 0.0,
                        }
                    )

            record.update(
                tile_availability(
                    context_mask,
                    current_area,
                    local_ids,
                    window,
                    full_height,
                    full_width,
                )
            )
            rating, warnings = availability_rating(
                record["candidate_positive_tiles"],
                record["candidate_negative_tiles"],
            )
            if record["context_flood_percentage"] < 0.5:
                warnings.append("local flood prevalence below 0.5%")
            if record["context_flood_percentage"] > 40.0:
                warnings.append("local flood prevalence above 40%")
            record["availability_rating"] = rating
            record["warnings"] = warnings
            records.append(record)

            print(
                f"  flood={official_cells:,} cells | context={context_cells:,} cells | "
                f"tiles +{record['candidate_positive_tiles']} / -{record['candidate_negative_tiles']} | {rating.upper()}"
            )
    finally:
        for dataset in datasets.values():
            dataset.close()

    area_frame = pd.DataFrame(records).sort_values("area_id")
    land_cover_frame = pd.DataFrame(land_cover_records)
    recommendation = build_recommendation(records)

    area_csv = OUTPUT_ROOT / "v3_area_audit.csv"
    area_json = OUTPUT_ROOT / "v3_area_audit.json"
    land_cover_csv = OUTPUT_ROOT / "v3_land_cover_distribution.csv"
    recommendation_json = OUTPUT_ROOT / "v3_candidate_split_recommendation.json"
    figure_path = FIGURE_ROOT / "v3_area_audit.png"

    csv_frame = area_frame.copy()
    csv_frame["warnings"] = csv_frame["warnings"].apply(lambda values: "; ".join(values))
    csv_frame.to_csv(area_csv, index=False)
    land_cover_frame.to_csv(land_cover_csv, index=False)

    audit_payload = {
        "audit": "ClimateAdapt V3 full-region training-area audit",
        "status": "completed",
        "created_before_v3_training": True,
        "v1_v2_prediction_outputs_accessed": False,
        "grid": {
            "crs": metadata.get("crs"),
            "resolution_metres": resolution,
            "width": full_width,
            "height": full_height,
        },
        "settings": {
            "context_metres": CONTEXT_METRES,
            "tile_size_pixels": TILE_SIZE,
            "stride_pixels": STRIDE,
            "minimum_context_fraction": MIN_CONTEXT_FRACTION,
        },
        "areas": records,
    }
    area_json.write_text(json.dumps(audit_payload, indent=2, default=json_default), encoding="utf-8")
    recommendation_json.write_text(json.dumps(recommendation, indent=2, default=json_default), encoding="utf-8")
    create_figure(area_frame, figure_path)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("V3 TRAINING-AREA AUDIT COMPLETED")
    print("=" * 78)
    for record in records:
        print(
            f"{record['coastal_area']:<25} "
            f"flood={record['official_flood_area_km2']:>7.3f} km²  "
            f"tiles +{record['candidate_positive_tiles']:>3}/-{record['candidate_negative_tiles']:<3}  "
            f"{record['availability_rating'].upper()}"
        )
    print("-" * 78)
    print(f"Area audit CSV:       {area_csv.relative_to(PROJECT_ROOT)}")
    print(f"Area audit JSON:      {area_json.relative_to(PROJECT_ROOT)}")
    print(f"Land-cover CSV:       {land_cover_csv.relative_to(PROJECT_ROOT)}")
    print(f"Split recommendation: {recommendation_json.relative_to(PROJECT_ROOT)}")
    print(f"Evidence figure:      {figure_path.relative_to(PROJECT_ROOT)}")
    print(f"Processing time:      {elapsed:.2f} seconds")
    print("=" * 78)
    print("No V3 split has been frozen and no model has been trained.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAudit cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
