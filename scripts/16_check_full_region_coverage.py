from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from shapely.geometry import box
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "nelson_region_full"
DEM_ROOT = RAW_ROOT / "dem"
LAND_COVER_ROOT = RAW_ROOT / "land_cover"
COASTLINE_PATH = RAW_ROOT / "coastline" / "nz_coastline_nelson_full.gpkg"
LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "labels"
    / "ncc_coastal_2050_ssp2_4_5_1pct_connected.geojson"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_region_coverage"
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures"
EXPECTED_CRS = "EPSG:2193"
FULL_THRESHOLD = 95.0


def first_layer(path: Path) -> str:
    layers = gpd.list_layers(path)
    if layers.empty:
        raise ValueError(f"No readable layer found in {path}")
    return str(layers.iloc[0]["name"])


def load_vector(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    layer = first_layer(path)
    data = gpd.read_file(path, layer=layer)
    if data.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    return data.to_crs(EXPECTED_CRS)


def combine_land_cover(paths: list[Path]) -> gpd.GeoDataFrame:
    frames = []
    for path in paths:
        data = load_vector(path)
        data["source_file"] = path.name
        frames.append(data)
    if not frames:
        raise FileNotFoundError(f"No GeoPackages found in {LAND_COVER_ROOT}")
    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=EXPECTED_CRS)


def inspect_dem_files(paths: list[Path]) -> tuple[list[dict], object]:
    records: list[dict] = []
    footprints = []

    for path in paths:
        with rasterio.open(path) as src:
            record = {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "crs": str(src.crs),
                "resolution_x": float(src.res[0]),
                "resolution_y": float(src.res[1]),
                "width": int(src.width),
                "height": int(src.height),
                "bands": int(src.count),
                "dtype": str(src.dtypes[0]),
                "nodata": None if src.nodata is None else float(src.nodata),
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
            }
            records.append(record)
            footprints.append(box(*src.bounds))

            if str(src.crs) != EXPECTED_CRS:
                raise ValueError(f"Unexpected CRS in {path}: {src.crs}")
            if not (
                abs(src.res[0] - 1.0) < 1e-6
                and abs(src.res[1] - 1.0) < 1e-6
            ):
                raise ValueError(f"Unexpected DEM resolution in {path}: {src.res}")
            if src.count != 1:
                raise ValueError(f"Expected one DEM band in {path}, found {src.count}")

    return records, unary_union(footprints)


def choose_area_field(labels: gpd.GeoDataFrame) -> str:
    preferred = ["InundationCellName", "inundationcellname", "name", "Name"]
    for column in preferred:
        if column in labels.columns:
            return column
    raise KeyError(
        "Could not find the coastal-area name column. Available columns: "
        + ", ".join(labels.columns)
    )


def calculate_coverage(labels: gpd.GeoDataFrame, dem_footprint) -> gpd.GeoDataFrame:
    area_field = choose_area_field(labels)
    dissolved = labels[[area_field, "geometry"]].dissolve(by=area_field).reset_index()
    dissolved = dissolved.rename(columns={area_field: "coastal_area"})

    percentages = []
    statuses = []
    for geometry in dissolved.geometry:
        total_area = geometry.area
        covered_area = geometry.intersection(dem_footprint).area
        percentage = 0.0 if total_area <= 0 else 100.0 * covered_area / total_area
        percentages.append(percentage)
        if percentage >= FULL_THRESHOLD:
            statuses.append("FULL")
        elif percentage > 0:
            statuses.append("PARTIAL")
        else:
            statuses.append("NONE")

    dissolved["coverage_percentage"] = percentages
    dissolved["status"] = statuses
    return dissolved


def create_figure(coverage: gpd.GeoDataFrame, dem_footprint, output_path: Path) -> None:
    colour_map = {"FULL": "#188038", "PARTIAL": "#f9ab00", "NONE": "#bdbdbd"}
    colours = coverage["status"].map(colour_map)

    figure, axis = plt.subplots(figsize=(12, 11))
    gpd.GeoSeries([dem_footprint], crs=EXPECTED_CRS).boundary.plot(
        ax=axis, color="#1a73e8", linewidth=2.0
    )
    coverage.plot(ax=axis, color=colours, edgecolor="#555555", linewidth=0.8)

    for row in coverage.itertuples():
        point = row.geometry.representative_point()
        axis.annotate(
            f"{row.coastal_area}\n{row.coverage_percentage:.1f}%",
            (point.x, point.y),
            fontsize=8,
            ha="center",
        )

    axis.set_title("ClimateAdapt AI – Full Nelson NCC Label Coverage", fontsize=16)
    axis.set_xlabel("Easting (NZTM2000)")
    axis.set_ylabel("Northing (NZTM2000)")
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)
    axis.legend(
        handles=[
            Patch(facecolor=colour_map["FULL"], label="Fully covered (≥95%)"),
            Patch(facecolor=colour_map["PARTIAL"], label="Partially covered"),
            Patch(facecolor=colour_map["NONE"], label="Not covered"),
            Patch(facecolor="none", edgecolor="#1a73e8", label="Combined DEM boundary"),
        ],
        loc="best",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    dem_paths = sorted(DEM_ROOT.rglob("*.tif"))
    land_cover_paths = sorted(LAND_COVER_ROOT.glob("*.gpkg"))
    if not dem_paths:
        raise FileNotFoundError(f"No DEM files found under {DEM_ROOT}")

    print("\nFULL-REGION SOURCE INVENTORY")
    print("=" * 80)
    print(f"DEM tiles:        {len(dem_paths)}")
    print(f"Land-cover files: {len(land_cover_paths)}")
    print(f"Coastline:        {COASTLINE_PATH.name}")
    print(f"Flood labels:     {LABEL_PATH.name}")

    dem_records, dem_footprint = inspect_dem_files(dem_paths)
    land_cover = combine_land_cover(land_cover_paths)
    coastline = load_vector(COASTLINE_PATH)
    labels = load_vector(LABEL_PATH)
    coverage = calculate_coverage(labels, dem_footprint)

    invalid_land_cover = int((~land_cover.geometry.is_valid).sum())
    invalid_coastline = int((~coastline.geometry.is_valid).sum())
    invalid_labels = int((~labels.geometry.is_valid).sum())

    print("\nVECTOR VALIDATION")
    print("=" * 80)
    print(f"Combined land-cover features: {len(land_cover):,}")
    print(f"Coastline features:           {len(coastline):,}")
    print(f"Connected flood features:     {len(labels):,}")
    print(f"Invalid land-cover shapes:    {invalid_land_cover}")
    print(f"Invalid coastline shapes:     {invalid_coastline}")
    print(f"Invalid flood-label shapes:   {invalid_labels}")

    print("\nNCC COASTAL-LABEL COVERAGE")
    print("=" * 80)
    display = coverage[["coastal_area", "coverage_percentage", "status"]].copy()
    display = display.sort_values(
        ["coverage_percentage", "coastal_area"], ascending=[False, True]
    )
    print(display.to_string(index=False, formatters={"coverage_percentage": "{:.2f}".format}))

    full_count = int((coverage["status"] == "FULL").sum())
    partial_count = int((coverage["status"] == "PARTIAL").sum())
    none_count = int((coverage["status"] == "NONE").sum())
    print("-" * 80)
    print(f"Fully covered:     {full_count}")
    print(f"Partially covered: {partial_count}")
    print(f"Not covered:       {none_count}")

    csv_path = OUTPUT_DIR / "full_region_label_coverage.csv"
    json_path = OUTPUT_DIR / "full_region_validation_summary.json"
    figure_path = FIGURE_DIR / "full_region_label_coverage.png"

    display.to_csv(csv_path, index=False)
    summary = {
        "expected_crs": EXPECTED_CRS,
        "dem_tile_count": len(dem_paths),
        "land_cover_file_count": len(land_cover_paths),
        "land_cover_feature_count_before_deduplication": len(land_cover),
        "coastline_feature_count": len(coastline),
        "label_feature_count": len(labels),
        "invalid_geometry_counts": {
            "land_cover": invalid_land_cover,
            "coastline": invalid_coastline,
            "labels": invalid_labels,
        },
        "coverage_counts": {
            "full": full_count,
            "partial": partial_count,
            "none": none_count,
        },
        "dem_tiles": dem_records,
        "coverage": display.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_figure(coverage, dem_footprint, figure_path)

    print("\nSAVED OUTPUTS")
    print("=" * 80)
    print(csv_path.relative_to(PROJECT_ROOT))
    print(json_path.relative_to(PROJECT_ROOT))
    print(figure_path.relative_to(PROJECT_ROOT))

    if none_count > 0 or partial_count > 0:
        print("\nRESULT: Additional DEM coverage is required before full-region preprocessing.")
    else:
        print("\nRESULT: Every connected NCC coastal-label area is fully covered.")


if __name__ == "__main__":
    main()
