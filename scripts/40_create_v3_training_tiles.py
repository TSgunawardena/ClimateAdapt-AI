"""Create leakage-controlled ClimateAdapt V3 U-Net tiles.

The script uses the frozen V3 geographic split. Normalisation is fitted only
on training pixels. Detailed LCDB codes are grouped into eight documented
semantic channels to avoid unseen-class failures. Validation and test tiles
are retained without label-based sampling; the test split is prepared
mechanically but is not evaluated.
"""

from __future__ import annotations

import hashlib
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RASTER_ROOT = PROJECT_ROOT / "data" / "processed" / "full_region" / "rasters"
V3_ROOT = PROJECT_ROOT / "data" / "processed" / "v3"
TILE_ROOT = V3_ROOT / "tiles"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "v3_audit"
FIGURE_ROOT = PROJECT_ROOT / "evidence" / "figures" / "v3"

PATHS = {
    "elevation": RASTER_ROOT / "elevation_5m.tif",
    "slope": RASTER_ROOT / "slope_degrees_5m.tif",
    "distance": RASTER_ROOT / "distance_to_coast_5m.tif",
    "land_cover": RASTER_ROOT / "land_cover_class_5m.tif",
    "flood_target": RASTER_ROOT / "flood_target_5m.tif",
    "valid": RASTER_ROOT / "valid_land_mask_5m.tif",
    "area_zone": V3_ROOT / "v3_area_zone_id_5m.tif",
    "split": V3_ROOT / "v3_split_mask_5m.tif",
    "split_metadata": V3_ROOT / "v3_split_metadata.json",
}

MANIFEST_PATH = TILE_ROOT / "v3_tile_manifest.csv"
METADATA_PATH = TILE_ROOT / "v3_tile_metadata.json"
DIST_FIGURE_PATH = FIGURE_ROOT / "v3_tile_distribution.png"
EXAMPLE_FIGURE_PATH = FIGURE_ROOT / "v3_example_training_tile.png"

TILE_SIZE = 128
STRIDE = 64
MIN_SPLIT_FRACTION = 0.20
TRAIN_NEGATIVE_TO_POSITIVE_RATIO = 1.0
RANDOM_SEED = 8532
LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995

SPLIT_NAMES = {1: "train", 2: "validation", 3: "test"}

# These are project-level engineering aggregations, not official LCDB classes.
LAND_COVER_GROUPS = {
    0: {"name": "unknown_or_unclassified", "source_codes": [0]},
    1: {"name": "developed_transport_or_extractive", "source_codes": [1, 2, 5, 6]},
    2: {"name": "bare_rock_sand_or_harvested", "source_codes": [10, 12, 16, 64]},
    3: {"name": "open_water", "source_codes": [20, 21, 22]},
    4: {"name": "agriculture_or_grassland", "source_codes": [30, 33, 40, 41, 43]},
    5: {"name": "wetland", "source_codes": [45, 46]},
    6: {"name": "shrub_or_scrub", "source_codes": [50, 51, 52, 55, 56, 58]},
    7: {"name": "forest_or_woodland", "source_codes": [54, 68, 69, 71]},
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raster(path: Path, floating: bool = False):
    with rasterio.open(path) as source:
        array = source.read(1)
        profile = source.profile.copy()
        nodata = source.nodata
    if floating:
        array = array.astype(np.float32, copy=False)
        if nodata is not None:
            array[array == nodata] = np.nan
    return array, profile


def verify_inputs() -> tuple[dict, dict]:
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        fail("Missing required inputs:\n  " + "\n  ".join(missing))

    split_metadata = json.loads(PATHS["split_metadata"].read_text(encoding="utf-8"))
    if split_metadata.get("status") != "frozen":
        fail("V3 split metadata is not marked frozen.")
    expected_hash = (
        split_metadata.get("output_hashes_sha256", {}).get("v3_split_mask_5m.tif")
    )
    if not expected_hash or sha256(PATHS["split"]) != expected_hash:
        fail("Frozen V3 split-mask checksum does not match its metadata.")

    reference = None
    reference_profile = None
    for label, path in PATHS.items():
        if label == "split_metadata":
            continue
        with rasterio.open(path) as source:
            signature = (source.width, source.height, str(source.crs), tuple(source.transform))
            if reference is None:
                reference = signature
                reference_profile = source.profile.copy()
            elif signature != reference:
                fail(f"Raster is not aligned: {label} ({path})")
    return split_metadata, reference_profile


def build_land_cover_lookup() -> np.ndarray:
    lookup = np.zeros(256, dtype=np.uint8)
    used_codes = set()
    for group_id, definition in LAND_COVER_GROUPS.items():
        for code in definition["source_codes"]:
            if code in used_codes:
                fail(f"LCDB code {code} appears in more than one semantic group.")
            used_codes.add(code)
            lookup[code] = np.uint8(group_id)
    return lookup


def calculate_normalisation(
    elevation: np.ndarray,
    slope: np.ndarray,
    distance_log: np.ndarray,
    train_mask: np.ndarray,
) -> dict:
    result = {}
    for name, array in [
        ("elevation", elevation),
        ("slope", slope),
        ("log1p_distance_to_coast", distance_log),
    ]:
        values = array[train_mask & np.isfinite(array)].astype(np.float64, copy=False)
        if values.size == 0:
            fail(f"No finite training values exist for {name}.")
        lower, upper = np.quantile(values, [LOWER_QUANTILE, UPPER_QUANTILE])
        clipped = np.clip(values, lower, upper)
        mean = float(clipped.mean())
        std = float(clipped.std())
        if std <= 1e-8:
            fail(f"Training standard deviation is zero for {name}.")
        result[name] = {
            "lower": float(lower),
            "upper": float(upper),
            "mean": mean,
            "std": std,
        }
        print(
            f"{name:<27} lower={lower:>9.3f} upper={upper:>9.3f} "
            f"mean={mean:>9.3f} std={std:>9.3f}"
        )
    return result


def normalise(array: np.ndarray, settings: dict) -> np.ndarray:
    clipped = np.clip(array, settings["lower"], settings["upper"])
    return ((clipped - settings["mean"]) / settings["std"]).astype(np.float32)


def dominant_positive(values: np.ndarray) -> int:
    values = values[values > 0]
    if values.size == 0:
        return 0
    counts = np.bincount(values.astype(np.int64))
    return int(np.argmax(counts))


def scan_candidates(
    split_mask: np.ndarray,
    valid_base: np.ndarray,
    flood_target: np.ndarray,
    area_zone: np.ndarray,
    area_names: dict[int, str],
) -> tuple[list[dict], dict]:
    height, width = split_mask.shape
    minimum_pixels = int(TILE_SIZE * TILE_SIZE * MIN_SPLIT_FRACTION)
    candidates = []
    rejected = Counter()

    for row in range(0, height - TILE_SIZE + 1, STRIDE):
        for column in range(0, width - TILE_SIZE + 1, STRIDE):
            rs = slice(row, row + TILE_SIZE)
            cs = slice(column, column + TILE_SIZE)
            split_tile = split_mask[rs, cs]
            split_values = np.unique(split_tile[split_tile > 0])
            if len(split_values) == 0:
                rejected["outside_split"] += 1
                continue
            if len(split_values) > 1:
                rejected["crosses_split_boundary"] += 1
                continue
            split_id = int(split_values[0])
            tile_mask = valid_base[rs, cs] & (split_tile == split_id)
            valid_pixels = int(tile_mask.sum())
            if valid_pixels < minimum_pixels:
                rejected["insufficient_split_pixels"] += 1
                continue
            tile_target = (flood_target[rs, cs] > 0) & tile_mask
            flood_pixels = int(tile_target.sum())
            area_id = dominant_positive(area_zone[rs, cs][tile_mask])
            candidates.append(
                {
                    "row": row,
                    "column": column,
                    "split_id": split_id,
                    "split": SPLIT_NAMES[split_id],
                    "area_id": area_id,
                    "coastal_area": area_names.get(area_id, "unknown"),
                    "valid_pixels": valid_pixels,
                    "flood_pixels": flood_pixels,
                    "contains_flood": flood_pixels > 0,
                }
            )
    return candidates, dict(rejected)


def select_candidates(candidates: list[dict]) -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    selected = []
    for split_name in ["train", "validation", "test"]:
        subset = [record for record in candidates if record["split"] == split_name]
        positive = [record for record in subset if record["contains_flood"]]
        negative = [record for record in subset if not record["contains_flood"]]
        if split_name == "train":
            maximum_negative = int(math.ceil(len(positive) * TRAIN_NEGATIVE_TO_POSITIVE_RATIO))
            if len(negative) > maximum_negative:
                chosen_indices = rng.choice(len(negative), size=maximum_negative, replace=False)
                negative = [negative[int(index)] for index in sorted(chosen_indices)]
        # Validation and test are never sampled according to their labels.
        selected.extend(positive)
        selected.extend(negative)
    return sorted(selected, key=lambda item: (item["split_id"], item["row"], item["column"]))


def make_input_tile(
    record: dict,
    elevation_n: np.ndarray,
    slope_n: np.ndarray,
    distance_n: np.ndarray,
    land_group: np.ndarray,
    valid_base: np.ndarray,
    split_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row, column = record["row"], record["column"]
    rs = slice(row, row + TILE_SIZE)
    cs = slice(column, column + TILE_SIZE)
    mask = valid_base[rs, cs] & (split_mask[rs, cs] == record["split_id"])
    channels = 3 + len(LAND_COVER_GROUPS)
    x = np.zeros((channels, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    x[0][mask] = elevation_n[rs, cs][mask]
    x[1][mask] = slope_n[rs, cs][mask]
    x[2][mask] = distance_n[rs, cs][mask]
    local_land = land_group[rs, cs]
    for group_id in LAND_COVER_GROUPS:
        x[3 + group_id][mask & (local_land == group_id)] = 1.0
    return x, mask.astype(np.uint8)


def clean_previous_tiles() -> None:
    for split_name in SPLIT_NAMES.values():
        folder = TILE_ROOT / split_name
        folder.mkdir(parents=True, exist_ok=True)
        for old_path in folder.glob("tile_*.npz"):
            old_path.unlink()


def create_figures(manifest: pd.DataFrame, example_path: Path) -> None:
    summary = []
    for split_name in ["train", "validation", "test"]:
        subset = manifest[manifest["split"] == split_name]
        summary.append(
            {
                "split": split_name.capitalize(),
                "positive": int(subset["contains_flood"].sum()),
                "negative": int((~subset["contains_flood"]).sum()),
            }
        )
    summary_frame = pd.DataFrame(summary)
    x_positions = np.arange(len(summary_frame))
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.bar(x_positions - 0.2, summary_frame["positive"], 0.4, label="Contains flood pixels", color="#dc2626")
    axis.bar(x_positions + 0.2, summary_frame["negative"], 0.4, label="No flood pixels", color="#4f83b3")
    axis.set_xticks(x_positions, summary_frame["split"])
    axis.set_ylabel("Number of tiles")
    axis.set_title("ClimateAdapt AI – V3 Tile Distribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(DIST_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    with np.load(example_path) as example:
        x = example["x"]
        y = example["y"]
        mask = example["valid_mask"]
        land = np.argmax(x[3:], axis=0)
    arrays = [x[0], x[1], x[2], land, y, mask]
    titles = [
        "Normalised elevation",
        "Normalised slope",
        "Normalised coast distance",
        "Grouped land cover",
        "Flood target",
        "Valid split mask",
    ]
    cmaps = ["viridis", "magma", "viridis", "tab10", "Reds", "gray"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("ClimateAdapt AI – Example V3 Positive Training Tile", fontsize=17, fontweight="bold")
    for axis, array, title, cmap in zip(axes.flat, arrays, titles, cmaps):
        axis.imshow(array, cmap=cmap)
        axis.set_title(title)
        axis.set_axis_off()
    fig.savefig(EXAMPLE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    TILE_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    split_metadata, reference_profile = verify_inputs()

    print("=" * 78)
    print("CLIMATEADAPT AI – V3 LEAKAGE-CONTROLLED TILE PREPARATION")
    print("=" * 78)
    print(f"Raster size: {reference_profile['width']:,} x {reference_profile['height']:,}")
    print(f"Tile size / stride: {TILE_SIZE} / {STRIDE}")
    print("Normalisation source: training split only")
    print("Test policy: mechanically prepared, not evaluated or used for selection")

    print("\nLoading aligned rasters...")
    elevation, _ = read_raster(PATHS["elevation"], floating=True)
    slope, _ = read_raster(PATHS["slope"], floating=True)
    distance, _ = read_raster(PATHS["distance"], floating=True)
    land_cover, _ = read_raster(PATHS["land_cover"])
    flood_target, _ = read_raster(PATHS["flood_target"])
    valid, _ = read_raster(PATHS["valid"])
    area_zone, _ = read_raster(PATHS["area_zone"])
    split_mask, _ = read_raster(PATHS["split"])

    finite_numeric = np.isfinite(elevation) & np.isfinite(slope) & np.isfinite(distance)
    valid_base = (valid == 1) & finite_numeric
    distance_log = np.log1p(np.maximum(distance, 0.0)).astype(np.float32)
    train_mask = valid_base & (split_mask == 1)

    print("\nCalculating training-only normalisation...")
    normalisation = calculate_normalisation(elevation, slope, distance_log, train_mask)
    elevation_n = normalise(elevation, normalisation["elevation"])
    slope_n = normalise(slope, normalisation["slope"])
    distance_n = normalise(distance_log, normalisation["log1p_distance_to_coast"])
    for array in [elevation_n, slope_n, distance_n]:
        array[~valid_base] = 0.0

    lookup = build_land_cover_lookup()
    source_codes = sorted(int(code) for code in np.unique(land_cover[valid_base]))
    unsupported_codes = [code for code in source_codes if code >= len(lookup) or code not in {
        item for definition in LAND_COVER_GROUPS.values() for item in definition["source_codes"]
    }]
    if unsupported_codes:
        fail(f"LCDB codes are not covered by the V3 semantic mapping: {unsupported_codes}")
    land_group = lookup[land_cover.astype(np.uint8)]

    area_assignments = split_metadata["configuration"]["area_assignments"]
    area_names = {int(area_id): details["coastal_area"] for area_id, details in area_assignments.items()}

    print("\nScanning frozen geographic regions for candidate tiles...")
    candidates, rejected = scan_candidates(split_mask, valid_base, flood_target, area_zone, area_names)
    selected = select_candidates(candidates)
    print(f"Candidate tiles before training balance: {len(candidates):,}")
    print(f"Selected tiles: {len(selected):,}")

    clean_previous_tiles()
    manifest_records = []
    example_positive = None
    print("\nSaving V3 U-Net tiles...")
    for index, record in enumerate(selected):
        row, column = record["row"], record["column"]
        rs = slice(row, row + TILE_SIZE)
        cs = slice(column, column + TILE_SIZE)
        x, tile_mask = make_input_tile(
            record, elevation_n, slope_n, distance_n, land_group, valid_base, split_mask
        )
        y = ((flood_target[rs, cs] > 0) & (tile_mask == 1)).astype(np.uint8)
        local_area = np.where(tile_mask == 1, area_zone[rs, cs], 0).astype(np.uint8)
        filename = f"tile_{index:05d}_r{row:04d}_c{column:04d}.npz"
        relative_path = Path(record["split"]) / filename
        output_path = TILE_ROOT / relative_path
        np.savez_compressed(
            output_path,
            x=x,
            y=y,
            valid_mask=tile_mask,
            area_zone_id=local_area,
            split_id=np.uint8(record["split_id"]),
            row=np.int32(row),
            column=np.int32(column),
        )
        saved = dict(record)
        saved.update(
            {
                "file": str(relative_path),
                "input_channels": int(x.shape[0]),
                "valid_pixels": int(tile_mask.sum()),
                "flood_pixels": int(y.sum()),
                "contains_flood": bool(y.any()),
            }
        )
        manifest_records.append(saved)
        if example_positive is None and record["split"] == "train" and y.any():
            example_positive = output_path

    manifest = pd.DataFrame(manifest_records)
    if manifest.empty:
        fail("No tiles were saved.")
    for split_name in ["train", "validation", "test"]:
        subset = manifest[manifest["split"] == split_name]
        if subset.empty or not subset["contains_flood"].any():
            fail(f"The {split_name} tile set is empty or has no positive tiles.")
    manifest.to_csv(MANIFEST_PATH, index=False)

    metadata = {
        "experiment": "ClimateAdapt V3",
        "status": "tiles_created_test_still_locked",
        "random_seed": RANDOM_SEED,
        "tile_size_pixels": TILE_SIZE,
        "stride_pixels": STRIDE,
        "resolution_metres": abs(float(reference_profile["transform"].a)),
        "tile_ground_extent_metres": TILE_SIZE * abs(float(reference_profile["transform"].a)),
        "minimum_split_fraction": MIN_SPLIT_FRACTION,
        "input_channels": 3 + len(LAND_COVER_GROUPS),
        "numeric_channels": ["elevation", "slope", "log1p_distance_to_coast"],
        "land_cover_semantic_groups": LAND_COVER_GROUPS,
        "land_cover_grouping_note": "Project engineering aggregation; not official LCDB class definitions.",
        "observed_source_land_cover_codes": source_codes,
        "normalisation": normalisation,
        "normalisation_fitted_on": "training pixels only",
        "training_negative_sampling_ratio": TRAIN_NEGATIVE_TO_POSITIVE_RATIO,
        "validation_sampling": "all eligible tiles retained",
        "test_sampling": "all eligible tiles retained without label-based sampling",
        "split_configuration_hash_sha256": split_metadata["configuration_hash_sha256"],
        "split_mask_sha256": sha256(PATHS["split"]),
        "candidate_tiles_before_training_balance": len(candidates),
        "saved_tiles": len(manifest),
        "rejected_candidate_windows": rejected,
        "test_access_policy": split_metadata["configuration"]["test_policy"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if example_positive is None:
        fail("No positive training tile was available for the evidence figure.")
    create_figures(manifest, example_positive)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("V3 TILE DATASET CREATED")
    print("=" * 78)
    for split_name in ["train", "validation", "test"]:
        subset = manifest[manifest["split"] == split_name]
        valid_pixels = int(subset["valid_pixels"].sum())
        flood_pixels = int(subset["flood_pixels"].sum())
        print(
            f"{split_name.capitalize():<12} tiles={len(subset):>4}  "
            f"positive={int(subset['contains_flood'].sum()):>4}  "
            f"negative={int((~subset['contains_flood']).sum()):>4}  "
            f"flood%={(100.0 * flood_pixels / valid_pixels if valid_pixels else 0):>6.2f}"
        )
    print("-" * 78)
    print(f"Input channels:      {3 + len(LAND_COVER_GROUPS)}")
    print(f"Manifest:            {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Metadata:            {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Distribution figure: {DIST_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Example figure:      {EXAMPLE_FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Processing time:     {elapsed:.2f} seconds")
    print("=" * 78)
    print("The locked test tiles were not evaluated and no model has been trained.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTile creation cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
