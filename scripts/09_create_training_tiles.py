from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


# =========================================================
# Configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RASTER_FOLDER = (
    PROJECT_ROOT
    / "data/processed/combined_rasters"
)

TILE_FOLDER = (
    PROJECT_ROOT
    / "data/processed/tiles"
)

EVIDENCE_FOLDER = PROJECT_ROOT / "evidence/figures"

TILE_FOLDER.mkdir(parents=True, exist_ok=True)
EVIDENCE_FOLDER.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 128
STRIDE = 64
MINIMUM_VALID_COVERAGE = 0.25
RANDOM_SEED = 42
NEGATIVE_TO_POSITIVE_RATIO = 1.0

SPLIT_NAMES = {
    1: "train",
    2: "validation",
    3: "test",
}

DEMONSTRATION_SPLIT_ID = 4
ALLOWED_SPLIT_IDS = {
    0,
    *SPLIT_NAMES.keys(),
    DEMONSTRATION_SPLIT_ID,
}


# =========================================================
# Helper: read raster
# =========================================================
def read_float_raster(path):
    with rasterio.open(path) as source:
        array = source.read(1).astype(np.float32)
        nodata = source.nodata
        profile = source.profile

    if nodata is not None:
        array[array == nodata] = np.nan

    return array, profile


def read_integer_raster(path):
    with rasterio.open(path) as source:
        array = source.read(1)
        profile = source.profile

    return array, profile


# =========================================================
# Step 1: Load combined rasters
# =========================================================
print("\nLoading combined rasters...")

elevation, reference_profile = read_float_raster(
    RASTER_FOLDER / "elevation_5m.tif"
)

slope, _ = read_float_raster(
    RASTER_FOLDER / "slope_degrees_5m.tif"
)

distance_to_coast, _ = read_float_raster(
    RASTER_FOLDER / "distance_to_coast_5m.tif"
)

land_cover, _ = read_integer_raster(
    RASTER_FOLDER / "land_cover_class_5m.tif"
)

flood_target, _ = read_integer_raster(
    RASTER_FOLDER / "flood_target_5m.tif"
)

valid_land_mask, _ = read_integer_raster(
    RASTER_FOLDER / "valid_land_mask_5m.tif"
)

split_mask, _ = read_integer_raster(
    RASTER_FOLDER / "split_mask_5m.tif"
)

flood_area_id, _ = read_integer_raster(
    RASTER_FOLDER / "flood_area_id_5m.tif"
)


expected_shape = elevation.shape

arrays_to_check = {
    "slope": slope,
    "distance": distance_to_coast,
    "land cover": land_cover,
    "flood target": flood_target,
    "valid mask": valid_land_mask,
    "split mask": split_mask,
    "flood area ID": flood_area_id,
}

for name, array in arrays_to_check.items():
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} shape {array.shape} does not match "
            f"the elevation shape {expected_shape}."
        )

observed_split_ids = {
    int(value)
    for value in np.unique(split_mask)
}

unexpected_split_ids = (
    observed_split_ids - ALLOWED_SPLIT_IDS
)

if unexpected_split_ids:
    raise ValueError(
        "Unexpected split ID(s) in split_mask_5m.tif: "
        f"{sorted(unexpected_split_ids)}. "
        "Expected 0=ignored, 1=train, 2=validation, "
        "3=test, or 4=demonstration."
    )

height, width = expected_shape

print(f"Raster size: {width} x {height}")
print(f"Tile size: {TILE_SIZE} x {TILE_SIZE}")
print(f"Stride: {STRIDE}")


# =========================================================
# Step 2: Calculate training-only normalization
# =========================================================
print("\nCalculating training-only normalization...")

training_pixels = (
    (valid_land_mask == 1)
    & (split_mask == 1)
)


def calculate_normalization(
    name,
    array,
    mask,
    use_log=False,
):
    working_array = array.copy()

    if use_log:
        working_array = np.log1p(
            np.maximum(working_array, 0)
        )

    values = working_array[
        mask & np.isfinite(working_array)
    ]

    if len(values) == 0:
        raise ValueError(
            f"No valid training values for {name}."
        )

    lower = float(np.percentile(values, 1))
    upper = float(np.percentile(values, 99))

    clipped_values = np.clip(
        values,
        lower,
        upper,
    )

    mean = float(clipped_values.mean())
    standard_deviation = float(
        clipped_values.std()
    )

    if standard_deviation < 1e-6:
        standard_deviation = 1.0

    normalized = np.clip(
        working_array,
        lower,
        upper,
    )

    normalized = (
        normalized - mean
    ) / standard_deviation

    normalized[
        ~np.isfinite(normalized)
    ] = 0.0

    statistics = {
        "name": name,
        "lower_percentile": lower,
        "upper_percentile": upper,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "log1p_transformation": use_log,
    }

    print(
        f"{name:<20} "
        f"lower={lower:>8.3f} "
        f"upper={upper:>8.3f} "
        f"mean={mean:>8.3f} "
        f"std={standard_deviation:>8.3f}"
    )

    return normalized.astype(np.float32), statistics


normalized_elevation, elevation_stats = (
    calculate_normalization(
        "elevation",
        elevation,
        training_pixels,
        use_log=False,
    )
)

normalized_slope, slope_stats = (
    calculate_normalization(
        "slope",
        slope,
        training_pixels,
        use_log=False,
    )
)

normalized_distance, distance_stats = (
    calculate_normalization(
        "distance_to_coast",
        distance_to_coast,
        training_pixels,
        use_log=True,
    )
)


continuous_features = np.stack(
    [
        normalized_elevation,
        normalized_slope,
        normalized_distance,
    ],
    axis=0,
)


# =========================================================
# Step 3: Determine land-cover classes
# =========================================================
land_cover_classes = sorted(
    int(value)
    for value in np.unique(
        land_cover[valid_land_mask == 1]
    )
)

print("\nLand-cover classes:")
print(land_cover_classes)

channel_names = [
    "elevation_normalized",
    "slope_normalized",
    "log_distance_to_coast_normalized",
]

for class_value in land_cover_classes:
    channel_names.append(
        f"land_cover_class_{class_value}"
    )

number_of_channels = len(channel_names)

print(f"Total U-Net input channels: {number_of_channels}")


# =========================================================
# Step 4: Generate possible tile locations
# =========================================================
def generate_positions(length):
    if length < TILE_SIZE:
        raise ValueError(
            f"Raster dimension {length} is smaller "
            f"than tile size {TILE_SIZE}."
        )

    positions = list(
        range(
            0,
            length - TILE_SIZE + 1,
            STRIDE,
        )
    )

    final_position = length - TILE_SIZE

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


row_positions = generate_positions(height)
column_positions = generate_positions(width)

candidate_tiles = []

for row in row_positions:
    for column in column_positions:
        row_end = row + TILE_SIZE
        column_end = column + TILE_SIZE

        valid_tile = valid_land_mask[
            row:row_end,
            column:column_end,
        ]

        split_tile = split_mask[
            row:row_end,
            column:column_end,
        ]

        split_values = np.unique(
            split_tile[
                (valid_tile == 1)
                & (split_tile > 0)
            ]
        )

        # Reject tiles crossing geographic split boundaries.
        if len(split_values) != 1:
            continue

        split_id = int(split_values[0])

        # Split 4 is the protected Waimea/Monaco demonstration
        # holdout. It must not enter model development tiles.
        if split_id == DEMONSTRATION_SPLIT_ID:
            continue

        # Defensive check: only model-development splits may
        # proceed to the manifest and tile folders.
        if split_id not in SPLIT_NAMES:
            continue

        region_mask = (
            (valid_tile == 1)
            & (split_tile == split_id)
        )

        valid_count = int(region_mask.sum())

        valid_coverage = (
            valid_count
            / (TILE_SIZE * TILE_SIZE)
        )

        if valid_coverage < MINIMUM_VALID_COVERAGE:
            continue

        target_tile = flood_target[
            row:row_end,
            column:column_end,
        ]

        flood_count = int(
            (
                (target_tile == 1)
                & region_mask
            ).sum()
        )

        candidate_tiles.append(
            {
                "row": row,
                "column": column,
                "split_id": split_id,
                "split": SPLIT_NAMES[split_id],
                "valid_count": valid_count,
                "valid_coverage": valid_coverage,
                "flood_count": flood_count,
                "contains_flood": flood_count > 0,
            }
        )


print(f"\nCandidate tiles: {len(candidate_tiles)}")


# =========================================================
# Step 5: Balance training tiles
# =========================================================
random_generator = np.random.default_rng(
    RANDOM_SEED
)

training_candidates = [
    tile
    for tile in candidate_tiles
    if tile["split_id"] == 1
]

positive_training_tiles = [
    tile
    for tile in training_candidates
    if tile["contains_flood"]
]

negative_training_tiles = [
    tile
    for tile in training_candidates
    if not tile["contains_flood"]
]

maximum_negative_tiles = int(
    len(positive_training_tiles)
    * NEGATIVE_TO_POSITIVE_RATIO
)

if len(negative_training_tiles) > maximum_negative_tiles:
    selected_indices = random_generator.choice(
        len(negative_training_tiles),
        size=maximum_negative_tiles,
        replace=False,
    )

    selected_negative_tiles = [
        negative_training_tiles[index]
        for index in selected_indices
    ]
else:
    selected_negative_tiles = negative_training_tiles

selected_training_tiles = (
    positive_training_tiles
    + selected_negative_tiles
)

selected_training_tiles.sort(
    key=lambda item: (
        item["row"],
        item["column"],
    )
)

validation_tiles = [
    tile
    for tile in candidate_tiles
    if tile["split_id"] == 2
]

test_tiles = [
    tile
    for tile in candidate_tiles
    if tile["split_id"] == 3
]

selected_tiles = (
    selected_training_tiles
    + validation_tiles
    + test_tiles
)


# =========================================================
# Step 6: Prepare tile folders
# =========================================================
for split_name in SPLIT_NAMES.values():
    split_folder = TILE_FOLDER / split_name
    split_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only old generated NPZ tiles
    for old_tile in split_folder.glob("tile_*.npz"):
        old_tile.unlink()


# =========================================================
# Step 7: Create and save tile arrays
# =========================================================
print("\nSaving U-Net tiles...")

manifest_records = []

for tile_number, tile in enumerate(selected_tiles):
    row = tile["row"]
    column = tile["column"]

    row_end = row + TILE_SIZE
    column_end = column + TILE_SIZE

    split_id = tile["split_id"]
    split_name = tile["split"]

    region_mask = (
        (
            valid_land_mask[
                row:row_end,
                column:column_end,
            ]
            == 1
        )
        & (
            split_mask[
                row:row_end,
                column:column_end,
            ]
            == split_id
        )
    )

    feature_channels = []

    # Three continuous channels
    for channel_index in range(
        continuous_features.shape[0]
    ):
        channel = continuous_features[
            channel_index,
            row:row_end,
            column:column_end,
        ].copy()

        channel[~region_mask] = 0.0
        feature_channels.append(channel)

    # One-hot land-cover channels
    land_cover_tile = land_cover[
        row:row_end,
        column:column_end,
    ]

    for class_value in land_cover_classes:
        class_channel = (
            (land_cover_tile == class_value)
            & region_mask
        ).astype(np.float32)

        feature_channels.append(class_channel)

    features = np.stack(
        feature_channels,
        axis=0,
    ).astype(np.float16)

    target = (
        (
            flood_target[
                row:row_end,
                column:column_end,
            ]
            == 1
        )
        & region_mask
    ).astype(np.uint8)

    mask = region_mask.astype(np.uint8)

    area_id = flood_area_id[
        row:row_end,
        column:column_end,
    ].copy()

    area_id[~region_mask] = 0

    filename = (
        f"tile_{tile_number:04d}"
        f"_r{row:04d}"
        f"_c{column:04d}.npz"
    )

    relative_path = (
        Path(split_name)
        / filename
    )

    output_path = TILE_FOLDER / relative_path

    np.savez_compressed(
        output_path,
        features=features,
        target=target,
        valid_mask=mask,
        flood_area_id=area_id.astype(np.uint8),
        row=np.int32(row),
        column=np.int32(column),
        split_id=np.uint8(split_id),
    )

    manifest_records.append(
        {
            "file": str(relative_path),
            "split": split_name,
            "row": row,
            "column": column,
            "valid_pixels": int(mask.sum()),
            "flood_pixels": int(target.sum()),
            "contains_flood": bool(target.sum() > 0),
        }
    )


# =========================================================
# Step 8: Save manifest and metadata
# =========================================================
manifest = pd.DataFrame(manifest_records)

manifest_path = TILE_FOLDER / "tile_manifest.csv"
manifest.to_csv(
    manifest_path,
    index=False,
)

normalization_metadata = {
    "tile_size_pixels": TILE_SIZE,
    "ground_size_metres": (
        TILE_SIZE * 5
    ),
    "stride_pixels": STRIDE,
    "stride_metres": STRIDE * 5,
    "minimum_valid_coverage": (
        MINIMUM_VALID_COVERAGE
    ),
    "random_seed": RANDOM_SEED,
    "negative_to_positive_ratio": (
        NEGATIVE_TO_POSITIVE_RATIO
    ),
    "number_of_channels": number_of_channels,
    "channel_names": channel_names,
    "land_cover_classes": land_cover_classes,
    "normalization": {
        "elevation": elevation_stats,
        "slope": slope_stats,
        "distance_to_coast": distance_stats,
    },
    "split_meaning": {
        "train": "Nelson CBD and covered Nelson Haven",
        "validation": "Rocks Road",
        "test": "Tahunanui",
    },
    "excluded_demonstration_split": {
        "split_id": DEMONSTRATION_SPLIT_ID,
        "area": "Waimea inlet and Monaco",
        "purpose": (
            "Protected unseen demonstration holdout; not tiled "
            "for training, validation, or independent testing."
        ),
    },
}

metadata_path = TILE_FOLDER / "tile_metadata.json"

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        normalization_metadata,
        file,
        indent=4,
    )

print(f"SAVED: {manifest_path.relative_to(PROJECT_ROOT)}")
print(f"SAVED: {metadata_path.relative_to(PROJECT_ROOT)}")


# =========================================================
# Step 9: Print tile statistics
# =========================================================
print("\n" + "=" * 74)
print("U-NET TILE DATASET")
print("=" * 74)

for split_name in [
    "train",
    "validation",
    "test",
]:
    split_records = manifest[
        manifest["split"] == split_name
    ]

    total_tiles = len(split_records)

    positive_tiles = int(
        split_records[
            "contains_flood"
        ].sum()
    )

    negative_tiles = (
        total_tiles - positive_tiles
    )

    total_valid_pixels = int(
        split_records[
            "valid_pixels"
        ].sum()
    )

    total_flood_pixels = int(
        split_records[
            "flood_pixels"
        ].sum()
    )

    flood_percentage = (
        (
            total_flood_pixels
            / total_valid_pixels
        )
        * 100
        if total_valid_pixels > 0
        else 0.0
    )

    print(
        f"{split_name.capitalize():<12} "
        f"tiles={total_tiles:>4}  "
        f"positive={positive_tiles:>4}  "
        f"negative={negative_tiles:>4}  "
        f"flood pixels={flood_percentage:>6.2f}%"
    )

print("-" * 74)
print(f"Input channels: {number_of_channels}")
print(
    f"Tile dimensions: "
    f"{TILE_SIZE} x {TILE_SIZE} pixels"
)
print(
    f"Ground dimensions: "
    f"{TILE_SIZE * 5} x "
    f"{TILE_SIZE * 5} metres"
)
print(
    "Demonstration split 4 (Waimea/Monaco): "
    "excluded from model-development tiles"
)
print("=" * 74)


# =========================================================
# Step 10: Create tile-distribution evidence
# =========================================================
summary = []

for split_name in [
    "train",
    "validation",
    "test",
]:
    split_records = manifest[
        manifest["split"] == split_name
    ]

    summary.append(
        {
            "split": split_name.capitalize(),
            "positive": int(
                split_records[
                    "contains_flood"
                ].sum()
            ),
            "negative": int(
                (
                    ~split_records[
                        "contains_flood"
                    ]
                ).sum()
            ),
        }
    )

summary_frame = pd.DataFrame(summary)

x_positions = np.arange(
    len(summary_frame)
)

bar_width = 0.36

figure, axis = plt.subplots(
    figsize=(9, 6)
)

axis.bar(
    x_positions - bar_width / 2,
    summary_frame["positive"],
    width=bar_width,
    label="Contains flood pixels",
    color="#d62728",
)

axis.bar(
    x_positions + bar_width / 2,
    summary_frame["negative"],
    width=bar_width,
    label="No flood pixels",
    color="#4c78a8",
)

axis.set_xticks(
    x_positions,
    summary_frame["split"],
)

axis.set_ylabel("Number of tiles")
axis.set_title(
    "ClimateAdapt AI – U-Net Tile Distribution",
    fontweight="bold",
)
axis.legend()
axis.grid(
    axis="y",
    alpha=0.25,
)

figure.tight_layout()

distribution_path = (
    EVIDENCE_FOLDER
    / "tile_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print(
    f"SAVED: "
    f"{distribution_path.relative_to(PROJECT_ROOT)}"
)


# =========================================================
# Step 11: Create example training-tile figure
# =========================================================
positive_training_manifest = manifest[
    (manifest["split"] == "train")
    & (manifest["contains_flood"])
]

if not positive_training_manifest.empty:
    example_record = (
        positive_training_manifest.iloc[0]
    )

    example_path = (
        TILE_FOLDER
        / example_record["file"]
    )

    example = np.load(example_path)

    example_features = example["features"].astype(
        np.float32
    )

    example_target = example["target"]
    example_mask = example["valid_mask"]

    example_figure, example_axes = plt.subplots(
        1,
        5,
        figsize=(17, 4),
    )

    example_images = [
        (
            example_features[0],
            "Normalized elevation",
            "terrain",
        ),
        (
            example_features[1],
            "Normalized slope",
            "magma",
        ),
        (
            example_features[2],
            "Normalized coast distance",
            "viridis",
        ),
        (
            example_target,
            "Flood target",
            "Reds",
        ),
        (
            example_mask,
            "Valid mask",
            "Greys",
        ),
    ]

    for axis, (
        image_data,
        title,
        colour_map,
    ) in zip(
        example_axes,
        example_images,
    ):
        axis.imshow(
            image_data,
            cmap=colour_map,
        )
        axis.set_title(title)
        axis.axis("off")

    example_figure.suptitle(
        "Example Positive Training Tile",
        fontweight="bold",
    )

    example_figure.tight_layout()

    example_figure_path = (
        EVIDENCE_FOLDER
        / "example_training_tile.png"
    )

    plt.savefig(
        example_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"SAVED: "
        f"{example_figure_path.relative_to(PROJECT_ROOT)}"
    )
