# Data handling rules

Keep original downloads unchanged in `raw`. Every transformed file goes into `interim` or `processed`.

Expected manual data:

- one or more GeoTIFF files in `raw/dem/` (begin with `tahunanui_dem_1m.tif`)
- `raw/land_cover/lcdb_v6_nelson.gpkg`
- `raw/coastline/nz_coastline_topo50.gpkg`

The label script creates:

- `raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_all.geojson`
- `raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_connected.geojson`
- `raw/labels/ncc_coastal_2050_ssp2_4_5_1pct_tahunanui_connected.geojson`
- `raw/labels/ncc_label_metadata.json`

Check every layer in QGIS before preprocessing. Confirm CRS, visible coverage, geometry validity, missing values, and whether Tahunanui is present.
