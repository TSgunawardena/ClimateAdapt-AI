from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

try:
    from shapely import make_valid as shapely_make_valid
except ImportError:  # Shapely < 2 fallback
    shapely_make_valid = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FULL_ROOT = PROJECT_ROOT / "data" / "raw" / "nelson_region_full"
RAW_LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "labels"
    / "ncc_coastal_2050_ssp2_4_5_1pct_connected.geojson"
)

LAND_COVER_ROOT = RAW_FULL_ROOT / "land_cover"
COASTLINE_PATH = RAW_FULL_ROOT / "coastline" / "nz_coastline_nelson_full.gpkg"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "full_region" / "vectors"
OUTPUT_LAND_COVER = OUTPUT_ROOT / "lcdb_nelson_combined_valid.gpkg"
OUTPUT_COASTLINE = OUTPUT_ROOT / "coastline_nelson_valid.gpkg"
OUTPUT_LABELS = OUTPUT_ROOT / "ncc_coastal_labels_valid.gpkg"

REPORT_ROOT = PROJECT_ROOT / "outputs" / "full_region_vectors"
SUMMARY_PATH = REPORT_ROOT / "vector_cleaning_summary.json"
CSV_PATH = REPORT_ROOT / "vector_cleaning_summary.csv"
FIGURE_PATH = PROJECT_ROOT / "evidence" / "figures" / "full_region_cleaned_vectors.png"

EXPECTED_CRS = "EPSG:2193"


def first_layer(path: Path) -> str:
    layers = gpd.list_layers(path)
    if layers.empty:
        raise ValueError(f"No readable layers in {path}")
    return str(layers.iloc[0]["name"])


def read_vector(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = gpd.read_file(path, layer=first_layer(path))
    if data.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    return data.to_crs(EXPECTED_CRS)


def flatten_polygon_parts(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    if isinstance(geometry, GeometryCollection):
        parts: list[Polygon] = []
        for item in geometry.geoms:
            parts.extend(flatten_polygon_parts(item))
        return parts
    return []


def repair_polygon(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if geometry.is_valid:
        repaired = geometry
    elif shapely_make_valid is not None:
        repaired = shapely_make_valid(geometry)
    else:
        repaired = geometry.buffer(0)

    parts = flatten_polygon_parts(repaired)
    if not parts:
        return None
    polygonal = unary_union(parts)
    if not polygonal.is_valid:
        polygonal = polygonal.buffer(0)
    return polygonal if not polygonal.is_empty else None


def repair_frame(data: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    before_count = len(data)
    invalid_before = int((~data.geometry.is_valid).sum())
    empty_before = int(data.geometry.is_empty.sum())

    repaired = data.copy()
    repaired.geometry = repaired.geometry.apply(repair_polygon)
    repaired = repaired[repaired.geometry.notna()].copy()
    repaired = repaired[~repaired.geometry.is_empty].copy()
    repaired = repaired[
        repaired.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    repaired = repaired.reset_index(drop=True)

    statistics = {
        "features_before": before_count,
        "features_after_repair": len(repaired),
        "invalid_before": invalid_before,
        "invalid_after": int((~repaired.geometry.is_valid).sum()),
        "empty_before": empty_before,
        "empty_after": int(repaired.geometry.is_empty.sum()),
    }
    return repaired, statistics


def find_column(data: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    lookup = {column.casefold(): column for column in data.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def prepare_land_cover() -> tuple[gpd.GeoDataFrame, dict]:
    paths = sorted(LAND_COVER_ROOT.glob("*.gpkg"))
    if not paths:
        raise FileNotFoundError(f"No LCDB GeoPackages found in {LAND_COVER_ROOT}")

    frames = []
    source_counts = {}
    for path in paths:
        frame = read_vector(path)
        frame["source_file"] = path.name
        source_counts[path.name] = len(frame)
        frames.append(frame)

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=EXPECTED_CRS
    )
    repaired, statistics = repair_frame(combined)

    uid_column = find_column(repaired, ["LCDB_UID"])
    if uid_column:
        missing_uid = repaired[uid_column].isna()
        with_uid = repaired[~missing_uid].copy()
        without_uid = repaired[missing_uid].copy()
        dissolved = with_uid.dissolve(by=uid_column, aggfunc="first", as_index=False)
        cleaned = gpd.GeoDataFrame(
            pd.concat([dissolved, without_uid], ignore_index=True),
            geometry="geometry",
            crs=EXPECTED_CRS,
        )
        deduplication_method = f"dissolved overlapping exports by {uid_column}"
    else:
        repaired["_geometry_wkb"] = repaired.geometry.to_wkb(hex=True)
        cleaned = repaired.drop_duplicates(subset=["_geometry_wkb"]).drop(
            columns=["_geometry_wkb"]
        )
        deduplication_method = "removed exact duplicate geometries"

    cleaned, final_statistics = repair_frame(cleaned)
    statistics.update(
        {
            "source_files": [path.name for path in paths],
            "source_feature_counts": source_counts,
            "features_after_deduplication": len(cleaned),
            "duplicates_or_overlaps_merged": len(repaired) - len(cleaned),
            "deduplication_method": deduplication_method,
            "final_invalid": final_statistics["invalid_after"],
        }
    )
    return cleaned, statistics


def prepare_coastline() -> tuple[gpd.GeoDataFrame, dict]:
    source = read_vector(COASTLINE_PATH)
    cleaned, statistics = repair_frame(source)
    statistics["source_files"] = [COASTLINE_PATH.name]
    statistics["features_after_deduplication"] = len(cleaned)
    statistics["duplicates_or_overlaps_merged"] = 0
    statistics["deduplication_method"] = "single authoritative export"
    statistics["final_invalid"] = int((~cleaned.geometry.is_valid).sum())
    return cleaned, statistics


def prepare_labels() -> tuple[gpd.GeoDataFrame, dict]:
    source = read_vector(RAW_LABEL_PATH)
    repaired, statistics = repair_frame(source)

    area_column = find_column(repaired, ["InundationCellName", "name"])
    if area_column is None:
        raise KeyError(
            "Could not find the NCC coastal-area field. Columns: "
            + ", ".join(repaired.columns)
        )

    cleaned = repaired.dissolve(by=area_column, aggfunc="first", as_index=False)
    cleaned, final_statistics = repair_frame(cleaned)
    statistics.update(
        {
            "source_files": [RAW_LABEL_PATH.name],
            "area_name_field": area_column,
            "features_after_deduplication": len(cleaned),
            "duplicates_or_overlaps_merged": len(repaired) - len(cleaned),
            "deduplication_method": f"dissolved by {area_column}",
            "final_invalid": final_statistics["invalid_after"],
        }
    )
    return cleaned, statistics


def save_geopackage(data: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    data.to_file(path, layer=layer, driver="GPKG", index=False)


def create_figure(
    land_cover: gpd.GeoDataFrame,
    coastline: gpd.GeoDataFrame,
    labels: gpd.GeoDataFrame,
) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 7))

    land_cover.plot(ax=axes[0], color="#b8d8b8", edgecolor="#6c8f6c", linewidth=0.15)
    axes[0].set_title("Combined and deduplicated LCDB")

    coastline.plot(ax=axes[1], color="#a8d5e5", edgecolor="#286c85", linewidth=0.35)
    axes[1].set_title("Validated coastline polygons")

    labels.plot(ax=axes[2], color="#e57373", edgecolor="#8b1e1e", linewidth=0.7)
    area_column = find_column(labels, ["InundationCellName", "name"])
    if area_column:
        for row in labels.itertuples():
            point = row.geometry.representative_point()
            label = getattr(row, area_column)
            axes[2].annotate(str(label), (point.x, point.y), fontsize=7, ha="center")
    axes[2].set_title("Repaired NCC coastal labels")

    for axis in axes:
        axis.set_xlabel("Easting (NZTM2000)")
        axis.set_ylabel("Northing (NZTM2000)")
        axis.set_aspect("equal")
        axis.grid(alpha=0.12)

    figure.suptitle("ClimateAdapt AI – Full-Region Vector Cleaning", fontsize=17)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("CLIMATEADAPT AI – FULL-REGION VECTOR CLEANING")
    print("=" * 80)
    print("Raw source files will not be modified.")

    print("\nPreparing LCDB land cover...")
    land_cover, land_cover_stats = prepare_land_cover()

    print("Preparing coastline polygons...")
    coastline, coastline_stats = prepare_coastline()

    print("Repairing NCC coastal labels...")
    labels, label_stats = prepare_labels()

    save_geopackage(land_cover, OUTPUT_LAND_COVER, "lcdb_nelson_combined_valid")
    save_geopackage(coastline, OUTPUT_COASTLINE, "coastline_nelson_valid")
    save_geopackage(labels, OUTPUT_LABELS, "ncc_coastal_labels_valid")

    records = []
    summaries = {
        "land_cover": land_cover_stats,
        "coastline": coastline_stats,
        "flood_labels": label_stats,
    }
    for dataset, values in summaries.items():
        records.append(
            {
                "dataset": dataset,
                "features_before": values["features_before"],
                "invalid_before": values["invalid_before"],
                "features_after": values["features_after_deduplication"],
                "invalid_after": values["final_invalid"],
                "merged_or_removed": values["duplicates_or_overlaps_merged"],
            }
        )

    summary = {
        "crs": EXPECTED_CRS,
        "raw_sources_preserved": True,
        "datasets": summaries,
        "outputs": {
            "land_cover": str(OUTPUT_LAND_COVER.relative_to(PROJECT_ROOT)),
            "coastline": str(OUTPUT_COASTLINE.relative_to(PROJECT_ROOT)),
            "flood_labels": str(OUTPUT_LABELS.relative_to(PROJECT_ROOT)),
            "evidence_figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    create_figure(land_cover, coastline, labels)

    print("\n" + "=" * 80)
    print("VECTOR CLEANING COMPLETED")
    print("=" * 80)
    for record in records:
        print(
            f"{record['dataset']:<15} "
            f"features {record['features_before']:>6,} -> {record['features_after']:>6,} | "
            f"invalid {record['invalid_before']:>4} -> {record['invalid_after']:>4} | "
            f"merged/removed {record['merged_or_removed']:>5,}"
        )

    print("-" * 80)
    print(f"Land cover: {OUTPUT_LAND_COVER.relative_to(PROJECT_ROOT)}")
    print(f"Coastline:  {OUTPUT_COASTLINE.relative_to(PROJECT_ROOT)}")
    print(f"Labels:     {OUTPUT_LABELS.relative_to(PROJECT_ROOT)}")
    print(f"Summary:    {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"CSV:        {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure:     {FIGURE_PATH.relative_to(PROJECT_ROOT)}")

    if any(record["invalid_after"] for record in records):
        raise RuntimeError("One or more cleaned outputs still contain invalid geometries.")
    if len(labels) != 9:
        raise RuntimeError(f"Expected 9 repaired NCC coastal areas, found {len(labels)}")

    print("\nRESULT: All processed vector outputs are valid and ready for rasterisation.")


if __name__ == "__main__":
    main()
