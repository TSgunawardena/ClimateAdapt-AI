"""Create and freeze the ClimateAdapt V3 geographic data split.

The split is area-based rather than random:
  training   – Nelson CBD, Nelson Haven, Rocks Road, Tahunanui, Waimea inlet
  validation – Boulder Bank Glenduan, Delaware Bay estuary
  testing    – Delaware Open Coast, Oananga Bay

Each valid-land context cell is assigned to its nearest labelled coastal area.
Where contexts from different splits meet, a separation belt is removed to
reduce spatial leakage. This script creates masks and evidence only; it does
not create tiles, normalise features, train a model, or select a threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "processed" / "full_region" / "rasters"
V3_ROOT = PROJECT_ROOT / "data" / "processed" / "v3"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v3_audit"
FIGURE_ROOT = PROJECT_ROOT / "evidence" / "figures" / "v3"

METADATA_PATH = SOURCE_ROOT / "full_region_metadata.json"
AREA_ID_PATH = SOURCE_ROOT / "flood_area_id_5m.tif"
VALID_MASK_PATH = SOURCE_ROOT / "valid_land_mask_5m.tif"

AREA_ZONE_PATH = V3_ROOT / "v3_area_zone_id_5m.tif"
SPLIT_MASK_PATH = V3_ROOT / "v3_split_mask_5m.tif"
SPLIT_METADATA_PATH = V3_ROOT / "v3_split_metadata.json"
FROZEN_SPLIT_PATH = OUTPUT_ROOT / "v3_frozen_split.json"
SPLIT_STATS_PATH = OUTPUT_ROOT / "v3_split_statistics.csv"
AREA_STATS_PATH = OUTPUT_ROOT / "v3_split_area_statistics.csv"
FIGURE_PATH = FIGURE_ROOT / "v3_geographic_split.png"

CONTEXT_METRES = 750.0
SEPARATION_GAP_METRES = 320.0
TILE_SIZE_PIXELS = 128
STRIDE_PIXELS = 64

SPLIT_IDS = {"ignored": 0, "training": 1, "validation": 2, "testing": 3}
SPLIT_NAMES = {value: key for key, value in SPLIT_IDS.items()}

AREA_TO_SPLIT_NAME = {
    "Boulder Bank Glenduan": "validation",
    "Delaware Bay estuary": "validation",
    "Delaware Open Coast": "testing",
    "Nelson CBD": "training",
    "Nelson Haven": "training",
    "Oananga Bay": "testing",
    "Rocks Road": "training",
    "Tahunanui": "training",
    "Waimea inlet": "training",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate() -> tuple[dict, dict[int, str], dict]:
    for path in [METADATA_PATH, AREA_ID_PATH, VALID_MASK_PATH]:
        if not path.exists():
            fail(f"Required source does not exist: {path}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    raw_mapping = metadata.get("flood_area_id_mapping")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        fail("The source metadata has no flood_area_id_mapping.")

    name_to_id = {str(name): int(area_id) for name, area_id in raw_mapping.items()}
    expected_names = set(AREA_TO_SPLIT_NAME)
    observed_names = set(name_to_id)
    if expected_names != observed_names:
        fail(
            "The configured V3 areas do not match the source metadata. "
            f"Missing={sorted(expected_names - observed_names)}; "
            f"unexpected={sorted(observed_names - expected_names)}"
        )

    with rasterio.open(AREA_ID_PATH) as area_source, rasterio.open(VALID_MASK_PATH) as valid_source:
        area_signature = (
            area_source.width,
            area_source.height,
            str(area_source.crs),
            tuple(area_source.transform),
        )
        valid_signature = (
            valid_source.width,
            valid_source.height,
            str(valid_source.crs),
            tuple(valid_source.transform),
        )
        if area_signature != valid_signature:
            fail("Flood-area ID and valid-land rasters are not aligned.")
        profile = area_source.profile.copy()

    id_to_name = {area_id: name for name, area_id in name_to_id.items()}
    return metadata, dict(sorted(id_to_name.items())), profile


def current_configuration(id_to_name: dict[int, str], resolution: float) -> dict:
    area_assignments = {
        str(area_id): {
            "coastal_area": name,
            "split": AREA_TO_SPLIT_NAME[name],
            "split_id": SPLIT_IDS[AREA_TO_SPLIT_NAME[name]],
        }
        for area_id, name in sorted(id_to_name.items())
    }
    return {
        "version": "v3",
        "method": "nearest labelled coastal area within valid-land context",
        "crs": "EPSG:2193",
        "resolution_metres": resolution,
        "context_metres": CONTEXT_METRES,
        "separation_gap_metres_total": SEPARATION_GAP_METRES,
        "tile_size_pixels": TILE_SIZE_PIXELS,
        "tile_ground_extent_metres": TILE_SIZE_PIXELS * resolution,
        "stride_pixels": STRIDE_PIXELS,
        "area_assignments": area_assignments,
        "test_policy": {
            "status": "locked_after_this_script",
            "allowed_before_final_evaluation": [
                "define geographic mask",
                "verify non-empty data availability",
                "create immutable test tiles",
            ],
            "prohibited_before_final_evaluation": [
                "model selection",
                "hyperparameter tuning",
                "checkpoint selection",
                "threshold selection",
                "feature engineering based on test performance",
            ],
        },
    }


def protect_frozen_configuration(configuration: dict) -> None:
    if not FROZEN_SPLIT_PATH.exists():
        return
    existing = json.loads(FROZEN_SPLIT_PATH.read_text(encoding="utf-8"))
    old_configuration = existing.get("configuration")
    if old_configuration != configuration:
        fail(
            "A different V3 frozen split already exists. Refusing to overwrite it. "
            "Use a new experiment version if the split must change."
        )


def expanded_window(
    rows: np.ndarray,
    cols: np.ndarray,
    padding: int,
    height: int,
    width: int,
) -> Window:
    row_start = max(0, int(rows.min()) - padding)
    row_stop = min(height, int(rows.max()) + padding + 1)
    col_start = max(0, int(cols.min()) - padding)
    col_stop = min(width, int(cols.max()) + padding + 1)
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def assign_nearest_area(
    area_ids: np.ndarray,
    valid: np.ndarray,
    id_to_name: dict[int, str],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = area_ids.shape
    padding = int(math.ceil(CONTEXT_METRES / resolution))
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    area_zone = np.zeros((height, width), dtype=np.uint8)

    for area_id, area_name in id_to_name.items():
        seed = (area_ids == area_id) & valid
        rows, cols = np.nonzero(seed)
        if rows.size == 0:
            fail(f"No valid-land label cells exist for {area_name} (ID {area_id}).")

        window = expanded_window(rows, cols, padding, height, width)
        row0 = int(window.row_off)
        col0 = int(window.col_off)
        local_height = int(window.height)
        local_width = int(window.width)
        rs = slice(row0, row0 + local_height)
        cs = slice(col0, col0 + local_width)

        local_valid = valid[rs, cs]
        local_seed = (area_ids[rs, cs] == area_id) & local_valid
        distance_metres = distance_transform_edt(~local_seed).astype(np.float32) * resolution
        candidate = local_valid & (distance_metres <= CONTEXT_METRES)

        existing_distance = best_distance[rs, cs]
        existing_area = area_zone[rs, cs]
        # Ascending ID order supplies deterministic tie-breaking.
        update = candidate & (distance_metres < existing_distance)
        existing_distance[update] = distance_metres[update]
        existing_area[update] = np.uint8(area_id)

        print(
            f"  Area {area_id}: {area_name:<24} "
            f"context candidates={int(candidate.sum()):,}"
        )

    best_distance[~np.isfinite(best_distance)] = -1.0
    return area_zone, best_distance


def area_zone_to_split(area_zone: np.ndarray, id_to_name: dict[int, str]) -> np.ndarray:
    split_mask = np.zeros_like(area_zone, dtype=np.uint8)
    for area_id, area_name in id_to_name.items():
        split_id = SPLIT_IDS[AREA_TO_SPLIT_NAME[area_name]]
        split_mask[area_zone == area_id] = np.uint8(split_id)
    return split_mask


def apply_separation_gap(raw_split: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    boundary = np.zeros(raw_split.shape, dtype=bool)

    horizontal = (
        (raw_split[:, :-1] > 0)
        & (raw_split[:, 1:] > 0)
        & (raw_split[:, :-1] != raw_split[:, 1:])
    )
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal

    vertical = (
        (raw_split[:-1, :] > 0)
        & (raw_split[1:, :] > 0)
        & (raw_split[:-1, :] != raw_split[1:, :])
    )
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical

    if not np.any(boundary):
        return raw_split.copy(), boundary

    half_gap = SEPARATION_GAP_METRES / 2.0
    distance_from_boundary = distance_transform_edt(~boundary).astype(np.float32) * resolution
    gap_mask = (raw_split > 0) & (distance_from_boundary <= half_gap)
    final_split = raw_split.copy()
    final_split[gap_mask] = 0
    return final_split, gap_mask


def output_profile(reference_profile: dict) -> dict:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    return profile


def write_raster(path: Path, array: np.ndarray, profile: dict, description: str) -> None:
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array.astype(np.uint8, copy=False), 1)
        destination.set_band_description(1, description)
        destination.update_tags(
            experiment="ClimateAdapt V3",
            created_by="39_create_v3_geographic_split.py",
        )


def make_statistics(
    area_ids: np.ndarray,
    valid: np.ndarray,
    area_zone: np.ndarray,
    raw_split: np.ndarray,
    final_split: np.ndarray,
    gap_mask: np.ndarray,
    id_to_name: dict[int, str],
    resolution: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_area_km2 = resolution * resolution / 1_000_000.0
    split_records = []
    for split_id in [1, 2, 3]:
        split_name = SPLIT_NAMES[split_id]
        mask = final_split == split_id
        raw_mask = raw_split == split_id
        flood_mask = (area_ids > 0) & valid & mask
        split_records.append(
            {
                "split_id": split_id,
                "split": split_name,
                "cells_before_gap": int(raw_mask.sum()),
                "cells_after_gap": int(mask.sum()),
                "cells_removed_by_gap": int((raw_mask & gap_mask).sum()),
                "area_after_gap_km2": float(mask.sum() * cell_area_km2),
                "official_flood_cells_retained": int(flood_mask.sum()),
                "official_flood_area_retained_km2": float(flood_mask.sum() * cell_area_km2),
                "flood_percentage": float(100.0 * flood_mask.sum() / mask.sum()) if mask.any() else 0.0,
            }
        )

    area_records = []
    for area_id, area_name in id_to_name.items():
        split_name = AREA_TO_SPLIT_NAME[area_name]
        split_id = SPLIT_IDS[split_name]
        original_flood = (area_ids == area_id) & valid
        retained_flood = original_flood & (final_split == split_id)
        zone_before = area_zone == area_id
        zone_after = zone_before & (final_split == split_id)
        original_count = int(original_flood.sum())
        retained_count = int(retained_flood.sum())
        area_records.append(
            {
                "area_id": area_id,
                "coastal_area": area_name,
                "split_id": split_id,
                "split": split_name,
                "zone_cells_before_gap": int(zone_before.sum()),
                "zone_cells_after_gap": int(zone_after.sum()),
                "official_flood_cells_original": original_count,
                "official_flood_cells_retained": retained_count,
                "official_flood_retained_percentage": (
                    100.0 * retained_count / original_count if original_count else 0.0
                ),
            }
        )

    return pd.DataFrame(split_records), pd.DataFrame(area_records)


def create_figure(
    final_split: np.ndarray,
    area_ids: np.ndarray,
    valid: np.ndarray,
    area_stats: pd.DataFrame,
    output_path: Path,
) -> None:
    max_dimension = 1400
    step = max(1, int(math.ceil(max(final_split.shape) / max_dimension)))
    split_small = final_split[::step, ::step]
    labels_small = area_ids[::step, ::step]
    valid_small = valid[::step, ::step]

    display = np.where(valid_small, split_small, np.nan)
    cmap = ListedColormap(["#ffffff", "#2ca02c", "#ffbf00", "#d62728"])
    cmap.set_bad("#f5f5f5")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, (axis_map, axis_bar) = plt.subplots(
        1,
        2,
        figsize=(17, 9),
        gridspec_kw={"width_ratios": [1.45, 1]},
        constrained_layout=True,
    )
    fig.suptitle("ClimateAdapt AI – Frozen V3 Geographic Split", fontsize=20, fontweight="bold")
    axis_map.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
    axis_map.contour(labels_small > 0, levels=[0.5], colors="black", linewidths=0.45)
    axis_map.set_title("Area-level split with separation gaps")
    axis_map.set_axis_off()
    axis_map.legend(
        handles=[
            Patch(facecolor="#2ca02c", label="Training"),
            Patch(facecolor="#ffbf00", label="Validation"),
            Patch(facecolor="#d62728", label="Locked testing"),
            Patch(facecolor="#ffffff", edgecolor="#777777", label="Ignored / separation gap"),
        ],
        loc="lower right",
        framealpha=0.95,
    )

    ordered = area_stats.sort_values(["split_id", "area_id"])
    colors = ordered["split_id"].map({1: "#2ca02c", 2: "#ffbf00", 3: "#d62728"})
    axis_bar.barh(ordered["coastal_area"], ordered["zone_cells_after_gap"], color=colors)
    axis_bar.invert_yaxis()
    axis_bar.set_title("Usable context retained by coastal area")
    axis_bar.set_xlabel("Valid-land cells")
    axis_bar.grid(axis="x", alpha=0.25)

    fig.text(
        0.5,
        0.003,
        "Locked test areas must not guide feature, model, checkpoint or threshold selection.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    V3_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    metadata, id_to_name, reference_profile = load_and_validate()
    resolution = abs(float(reference_profile["transform"].a))
    configuration = current_configuration(id_to_name, resolution)
    protect_frozen_configuration(configuration)

    print("=" * 78)
    print("CLIMATEADAPT AI – CREATE AND FREEZE V3 GEOGRAPHIC SPLIT")
    print("=" * 78)
    print(f"Grid: {reference_profile['width']:,} x {reference_profile['height']:,}")
    print(f"Resolution: {resolution:.1f} metres")
    print(f"Area context: {CONTEXT_METRES:.0f} metres")
    print(f"Cross-split separation belt: {SEPARATION_GAP_METRES:.0f} metres total")
    print("Training: Nelson CBD, Nelson Haven, Rocks Road, Tahunanui, Waimea inlet")
    print("Validation: Boulder Bank Glenduan, Delaware Bay estuary")
    print("Locked test: Delaware Open Coast, Oananga Bay")
    print("\nLoading source masks...")

    with rasterio.open(AREA_ID_PATH) as source:
        area_ids = source.read(1).astype(np.uint8, copy=False)
    with rasterio.open(VALID_MASK_PATH) as source:
        valid = source.read(1) == 1

    print("Assigning valid-land context to the nearest coastal area...")
    area_zone, nearest_distance = assign_nearest_area(
        area_ids, valid, id_to_name, resolution
    )
    del nearest_distance

    raw_split = area_zone_to_split(area_zone, id_to_name)
    raw_split[~valid] = 0
    print("Applying the cross-split separation belt...")
    final_split, gap_mask = apply_separation_gap(raw_split, resolution)

    split_stats, area_stats = make_statistics(
        area_ids,
        valid,
        area_zone,
        raw_split,
        final_split,
        gap_mask,
        id_to_name,
        resolution,
    )

    if (split_stats["cells_after_gap"] <= 0).any():
        fail("At least one split is empty after applying the separation gap.")
    if (split_stats["official_flood_cells_retained"] <= 0).any():
        fail("At least one split has no retained official flood cells.")

    print("Writing V3 geographic masks...")
    profile = output_profile(reference_profile)
    write_raster(AREA_ZONE_PATH, area_zone, profile, "V3 nearest coastal-area zone ID")
    write_raster(SPLIT_MASK_PATH, final_split, profile, "V3 split: 0 ignored, 1 train, 2 validation, 3 test")

    split_stats.to_csv(SPLIT_STATS_PATH, index=False)
    area_stats.to_csv(AREA_STATS_PATH, index=False)

    output_hashes = {
        "v3_area_zone_id_5m.tif": file_sha256(AREA_ZONE_PATH),
        "v3_split_mask_5m.tif": file_sha256(SPLIT_MASK_PATH),
    }
    source_hashes = {
        "full_region_metadata.json": file_sha256(METADATA_PATH),
        "flood_area_id_5m.tif": file_sha256(AREA_ID_PATH),
        "valid_land_mask_5m.tif": file_sha256(VALID_MASK_PATH),
    }
    configuration_hash = canonical_hash(configuration)

    split_metadata = {
        "experiment": "ClimateAdapt V3",
        "status": "frozen",
        "configuration_hash_sha256": configuration_hash,
        "configuration": configuration,
        "source_hashes_sha256": source_hashes,
        "output_hashes_sha256": output_hashes,
        "split_statistics": split_stats.to_dict(orient="records"),
        "area_statistics": area_stats.to_dict(orient="records"),
        "important_note": (
            "Test labels were used only to define and verify the geographic test mask. "
            "They are prohibited from model, feature, checkpoint and threshold selection."
        ),
    }
    SPLIT_METADATA_PATH.write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    FROZEN_SPLIT_PATH.write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")

    print("Creating split evidence figure...")
    create_figure(final_split, area_ids, valid, area_stats, FIGURE_PATH)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("V3 GEOGRAPHIC SPLIT FROZEN")
    print("=" * 78)
    for row in split_stats.itertuples(index=False):
        print(
            f"{row.split.capitalize():<12} "
            f"cells={row.cells_after_gap:>10,}  "
            f"flood={row.official_flood_cells_retained:>8,}  "
            f"flood%={row.flood_percentage:>6.2f}  "
            f"gap removed={row.cells_removed_by_gap:>8,}"
        )
    print("-" * 78)
    print(f"Area zones:       {AREA_ZONE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Split mask:       {SPLIT_MASK_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Split metadata:   {SPLIT_METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Frozen split:     {FROZEN_SPLIT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Split statistics: {SPLIT_STATS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Area statistics:  {AREA_STATS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Evidence figure:  {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Configuration:    {configuration_hash[:16]}...")
    print(f"Processing time:  {elapsed:.2f} seconds")
    print("=" * 78)
    print("The V3 test split is now locked. No model has been trained.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSplit creation cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
