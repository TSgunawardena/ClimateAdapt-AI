from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_PATH = PROJECT_ROOT / "data/raw/dem/BQ26.tif"
LABEL_PATH = (
    PROJECT_ROOT
    / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_connected.geojson"
)

OUTPUT_FOLDER = PROJECT_ROOT / "data/interim"
EVIDENCE_FOLDER = PROJECT_ROOT / "evidence/figures"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
EVIDENCE_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Read DEM boundary
# ---------------------------------------------------------
with rasterio.open(DEM_PATH) as source:
    dem_crs = source.crs
    dem_bounds = source.bounds

dem_boundary = box(
    dem_bounds.left,
    dem_bounds.bottom,
    dem_bounds.right,
    dem_bounds.top,
)

dem_boundary_gdf = gpd.GeoDataFrame(
    {"name": ["Current DEM coverage"]},
    geometry=[dem_boundary],
    crs=dem_crs,
)


# ---------------------------------------------------------
# Read all connected NCC coastal labels
# ---------------------------------------------------------
labels = gpd.read_file(LABEL_PATH).to_crs(dem_crs)

required_column = "InundationCellName"

if required_column not in labels.columns:
    raise ValueError(
        f"Required column '{required_column}' was not found. "
        f"Available columns: {labels.columns.tolist()}"
    )


# Combine duplicate records belonging to the same coastal area
coastal_areas = labels.dissolve(
    by=required_column,
    as_index=False,
)

coverage_records = []
covered_geometries = []

for _, row in coastal_areas.iterrows():
    area_name = row[required_column]
    geometry = row.geometry

    original_area = geometry.area
    intersection = geometry.intersection(dem_boundary)
    covered_area = intersection.area

    if original_area > 0:
        coverage_percentage = (covered_area / original_area) * 100
    else:
        coverage_percentage = 0.0

    if coverage_percentage >= 95:
        status = "FULL"
    elif coverage_percentage >= 5:
        status = "PARTIAL"
    else:
        status = "NONE"

    coverage_records.append(
        {
            "coastal_area": area_name,
            "coverage_percentage": round(coverage_percentage, 2),
            "status": status,
            "original_area_m2": round(original_area, 2),
            "covered_area_m2": round(covered_area, 2),
        }
    )

    if not intersection.is_empty:
        covered_geometries.append(
            {
                "coastal_area": area_name,
                "status": status,
                "geometry": intersection,
            }
        )


# ---------------------------------------------------------
# Save coverage table
# ---------------------------------------------------------
coverage_table = pd.DataFrame(coverage_records)
coverage_table = coverage_table.sort_values(
    by="coverage_percentage",
    ascending=False,
)

csv_path = OUTPUT_FOLDER / "spatial_coverage_report.csv"
coverage_table.to_csv(csv_path, index=False)

print("\n" + "=" * 76)
print("CURRENT DEM COVERAGE OF NCC COASTAL LABEL AREAS")
print("=" * 76)

print(
    coverage_table[
        ["coastal_area", "coverage_percentage", "status"]
    ].to_string(index=False)
)

print("=" * 76)

full_areas = coverage_table[
    coverage_table["status"] == "FULL"
]["coastal_area"].tolist()

partial_areas = coverage_table[
    coverage_table["status"] == "PARTIAL"
]["coastal_area"].tolist()

print(f"\nFully covered areas: {len(full_areas)}")
for area in full_areas:
    print(f"  - {area}")

print(f"\nPartially covered areas: {len(partial_areas)}")
for area in partial_areas:
    print(f"  - {area}")

print(f"\nSaved report: {csv_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------
# Create evidence map
# ---------------------------------------------------------
figure, axis = plt.subplots(figsize=(11, 9))

coastal_areas.plot(
    ax=axis,
    facecolor="lightgrey",
    edgecolor="grey",
    linewidth=1,
)

dem_boundary_gdf.boundary.plot(
    ax=axis,
    color="blue",
    linewidth=2,
)

if covered_geometries:
    covered_gdf = gpd.GeoDataFrame(
        covered_geometries,
        geometry="geometry",
        crs=dem_crs,
    )

    full_covered = covered_gdf[covered_gdf["status"] == "FULL"]
    partial_covered = covered_gdf[covered_gdf["status"] == "PARTIAL"]

    if not full_covered.empty:
        full_covered.plot(
            ax=axis,
            facecolor="green",
            edgecolor="darkgreen",
            alpha=0.65,
        )

    if not partial_covered.empty:
        partial_covered.plot(
            ax=axis,
            facecolor="orange",
            edgecolor="darkorange",
            alpha=0.65,
        )


# Add coastal-area labels
for _, row in coastal_areas.iterrows():
    label_point = row.geometry.representative_point()

    axis.annotate(
        text=row[required_column],
        xy=(label_point.x, label_point.y),
        fontsize=7,
        ha="center",
        va="center",
    )


legend_items = [
    Patch(
        facecolor="green",
        edgecolor="darkgreen",
        label="Fully covered (95% or more)",
    ),
    Patch(
        facecolor="orange",
        edgecolor="darkorange",
        label="Partially covered",
    ),
    Patch(
        facecolor="lightgrey",
        edgecolor="grey",
        label="Not covered",
    ),
    Patch(
        facecolor="none",
        edgecolor="blue",
        linewidth=2,
        label="Current DEM boundary",
    ),
]

axis.legend(handles=legend_items, loc="best")
axis.set_title(
    "ClimateAdapt AI – NCC Label Coverage by Current DEM",
    fontsize=14,
    fontweight="bold",
)
axis.set_xlabel("Easting (NZTM2000)")
axis.set_ylabel("Northing (NZTM2000)")
axis.set_aspect("equal")

figure.tight_layout()

figure_path = EVIDENCE_FOLDER / "spatial_coverage_map.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved map: {figure_path.relative_to(PROJECT_ROOT)}")