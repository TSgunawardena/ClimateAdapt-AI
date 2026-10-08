from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_PATH = PROJECT_ROOT / "data/raw/dem/BQ26.tif"
LAND_COVER_PATH = PROJECT_ROOT / "data/raw/land_cover/lcdb_v6_nelson.gpkg"
COASTLINE_PATH = PROJECT_ROOT / "data/raw/coastline/nz_coastline_topo50.gpkg"
LABEL_PATH = (
    PROJECT_ROOT
    / "data/raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_tahunanui_connected.geojson"
)

OUTPUT_FOLDER = PROJECT_ROOT / "evidence/figures"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_FOLDER / "raw_data_overview.png"


# ---------------------------------------------------------
# Load DEM at a reduced display resolution
# ---------------------------------------------------------
with rasterio.open(DEM_PATH) as src:
    display_height = min(src.height, 1200)
    display_width = int(src.width * (display_height / src.height))

    dem = src.read(
        1,
        out_shape=(display_height, display_width),
    ).astype("float32")

    if src.nodata is not None:
        dem[dem == src.nodata] = np.nan

    bounds = src.bounds
    dem_crs = src.crs

extent = [
    bounds.left,
    bounds.right,
    bounds.bottom,
    bounds.top,
]

valid_values = dem[np.isfinite(dem)]

if len(valid_values) > 0:
    minimum_display = np.percentile(valid_values, 2)
    maximum_display = np.percentile(valid_values, 98)
else:
    minimum_display = None
    maximum_display = None


# ---------------------------------------------------------
# Load vector datasets
# ---------------------------------------------------------
land_cover = gpd.read_file(LAND_COVER_PATH).to_crs(dem_crs)
coastline = gpd.read_file(COASTLINE_PATH).to_crs(dem_crs)
flood_label = gpd.read_file(LABEL_PATH).to_crs(dem_crs)


# ---------------------------------------------------------
# Create plots
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: DEM, coastline and flood label
dem_image = axes[0].imshow(
    dem,
    extent=extent,
    origin="upper",
    cmap="terrain",
    vmin=minimum_display,
    vmax=maximum_display,
)

coastline.boundary.plot(
    ax=axes[0],
    color="blue",
    linewidth=1.2,
)

flood_label.boundary.plot(
    ax=axes[0],
    color="red",
    linewidth=2,
)

axes[0].set_title("Tāhunanui DEM and Flood Label")
axes[0].set_xlabel("Easting (NZTM2000)")
axes[0].set_ylabel("Northing (NZTM2000)")
axes[0].set_xlim(bounds.left, bounds.right)
axes[0].set_ylim(bounds.bottom, bounds.top)
axes[0].set_aspect("equal")

colour_bar = fig.colorbar(
    dem_image,
    ax=axes[0],
    fraction=0.046,
    pad=0.04,
)
colour_bar.set_label("Elevation (metres)")


# Plot 2: Land cover and flood label
land_cover.plot(
    ax=axes[1],
    column="Name_2023",
    categorical=True,
    cmap="tab20",
    legend=False,
    edgecolor="grey",
    linewidth=0.2,
)

coastline.boundary.plot(
    ax=axes[1],
    color="blue",
    linewidth=1.2,
)

flood_label.plot(
    ax=axes[1],
    facecolor="red",
    edgecolor="darkred",
    alpha=0.40,
    linewidth=1.5,
)

axes[1].set_title("Land Cover and Official Flood Label")
axes[1].set_xlabel("Easting (NZTM2000)")
axes[1].set_ylabel("Northing (NZTM2000)")
axes[1].set_xlim(bounds.left, bounds.right)
axes[1].set_ylim(bounds.bottom, bounds.top)
axes[1].set_aspect("equal")


# Shared legend
legend_items = [
    Line2D(
        [0],
        [0],
        color="blue",
        linewidth=2,
        label="Coastline",
    ),
    Line2D(
        [0],
        [0],
        color="red",
        linewidth=2,
        label="Official flood extent",
    ),
]

fig.legend(
    handles=legend_items,
    loc="lower center",
    ncol=2,
)

fig.suptitle(
    "ClimateAdapt AI – Raw Data Inspection",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved inspection figure to: {OUTPUT_PATH}")