"""Windowed V3 place analysis from frozen full-region susceptibility rasters."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import uuid

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.windows import Window, bounds as window_bounds
import requests


DEFAULT_THRESHOLD = 0.004
LOWER_REVIEW_BOUND = 0.002
UPPER_REVIEW_BOUND = 0.010

BUILTIN_PLACES = {
    "atawhai": ("Atawhai, Nelson", -41.2479, 173.3277),
    "nelson": ("Nelson CBD", -41.2746, 173.2839),
    "nelson cbd": ("Nelson CBD", -41.2746, 173.2839),
    "rocks road": ("Rocks Road, Nelson", -41.27375, 173.25802),
    "tahunanui": ("Tāhunanui, Nelson", -41.2844, 173.2450),
    "tāhunanui": ("Tāhunanui, Nelson", -41.2844, 173.2450),
    "stoke": ("Stoke, Nelson", -41.3120, 173.2320),
    "monaco": ("Monaco, Nelson", -41.30417, 173.21111),
}


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:60] or "selected-place"


def json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def normalise_place_query(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    value = re.sub(r",?\s+(nelson city|nelson|new zealand|nz|aotearoa)\s*$", "", value)
    return value.strip(" ,")


class PlaceAnalysisService:
    """Resolve a Nelson place and summarise frozen V3 regional outputs."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        raster_dir = self.root / "data" / "processed" / "full_region" / "rasters"
        inference_dir = self.root / "outputs" / "full_region_inference_v3"
        self.summary_path = inference_dir / "nelson_logistic_full_region_summary_v3.json"
        self.paths = {
            "score": inference_dir / "nelson_logistic_susceptibility_score_v3.tif",
            "binary": inference_dir / "nelson_logistic_susceptibility_v3.tif",
            "coverage": inference_dir / "nelson_logistic_coverage_v3.tif",
            "elevation": raster_dir / "elevation_5m.tif",
            "slope": raster_dir / "slope_degrees_5m.tif",
            "distance": raster_dir / "distance_to_coast_5m.tif",
            "land_cover": raster_dir / "land_cover_class_5m.tif",
            "valid_land": raster_dir / "valid_land_mask_5m.tif",
        }

    def required_files(self) -> list[Path]:
        return [*self.paths.values(), self.summary_path]

    def missing_files(self) -> list[Path]:
        return [path for path in self.required_files() if not path.exists()]

    def resolve_place(
        self,
        query: str,
        manual_latitude: float | None = None,
        manual_longitude: float | None = None,
    ) -> dict:
        if manual_latitude is not None and manual_longitude is not None:
            if not -90 <= manual_latitude <= 90 or not -180 <= manual_longitude <= 180:
                raise ValueError("Manual latitude or longitude is outside its valid range.")
            return {
                "query": query.strip() or "Manual coordinate",
                "display_name": query.strip() or "Manual coordinate",
                "latitude": float(manual_latitude),
                "longitude": float(manual_longitude),
                "source": "manual_coordinate",
            }

        raw_query = query.strip()
        normalised = normalise_place_query(raw_query)
        if not normalised:
            raise ValueError("Enter a place name or provide manual coordinates.")
        if normalised in BUILTIN_PLACES:
            name, latitude, longitude = BUILTIN_PLACES[normalised]
            return {
                "query": raw_query,
                "display_name": name,
                "latitude": latitude,
                "longitude": longitude,
                "source": "offline_nelson_catalogue",
            }

        search_query = raw_query
        if "new zealand" not in raw_query.lower() and "nz" not in raw_query.lower():
            search_query = f"{raw_query}, Nelson, New Zealand"
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": search_query,
                    "format": "jsonv2",
                    "limit": 10,
                    "countrycodes": "nz",
                    "viewbox": "172.90,-41.60,173.70,-41.00",
                    "bounded": 1,
                },
                headers={"User-Agent": "ClimateAdaptAI-NMIT-Academic-Prototype/1.0"},
                timeout=10,
            )
            response.raise_for_status()
            candidates = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(
                "Online place search was unavailable. Select a built-in Nelson "
                "place or use manual latitude/longitude."
            ) from error

        query_tokens = {token for token in re.findall(r"[a-z0-9]+", normalised) if len(token) > 2}
        accepted = []
        for candidate in candidates:
            latitude = float(candidate["lat"])
            longitude = float(candidate["lon"])
            display_name = str(candidate.get("display_name", ""))
            display_tokens = set(re.findall(r"[a-z0-9]+", display_name.lower()))
            if not (-41.60 <= latitude <= -41.00 and 172.90 <= longitude <= 173.70):
                continue
            token_matches = len(query_tokens & display_tokens)
            if query_tokens and token_matches == 0:
                continue
            accepted.append((token_matches, candidate))

        if not accepted:
            raise ValueError(
                f"No trustworthy Nelson-region match was found for '{raw_query}'. "
                "Use a built-in location or manual coordinates."
            )
        accepted.sort(key=lambda item: item[0], reverse=True)
        candidate = accepted[0][1]
        return {
            "query": raw_query,
            "display_name": str(candidate.get("display_name", raw_query)),
            "latitude": float(candidate["lat"]),
            "longitude": float(candidate["lon"]),
            "source": "OpenStreetMap Nominatim – Nelson bounded",
        }

    def analyse(self, place: dict, radius_metres: int) -> dict:
        if radius_metres not in {500, 1000, 2000}:
            raise ValueError("Radius must be 500, 1000 or 2000 metres.")
        missing = self.missing_files()
        if missing:
            raise FileNotFoundError(
                "Missing required V3 regional outputs: "
                + ", ".join(str(path.relative_to(self.root)) for path in missing)
            )

        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        threshold = float(summary.get("frozen_threshold", DEFAULT_THRESHOLD))
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
        easting, northing = transformer.transform(
            float(place["longitude"]), float(place["latitude"])
        )

        with rasterio.open(self.paths["score"]) as reference:
            if reference.crs is None or reference.crs.to_string() != "EPSG:2193":
                raise ValueError("Expected V3 regional outputs in EPSG:2193.")
            resolution_x = abs(float(reference.transform.a))
            resolution_y = abs(float(reference.transform.e))
            column_centre, row_centre = (~reference.transform) * (easting, northing)
            half_columns = int(math.ceil(radius_metres / resolution_x)) + 1
            half_rows = int(math.ceil(radius_metres / resolution_y)) + 1
            column_start = max(0, int(math.floor(column_centre)) - half_columns)
            column_stop = min(reference.width, int(math.floor(column_centre)) + half_columns + 1)
            row_start = max(0, int(math.floor(row_centre)) - half_rows)
            row_stop = min(reference.height, int(math.floor(row_centre)) + half_rows + 1)
            if column_start >= column_stop or row_start >= row_stop:
                return self._unavailable(place, radius_metres, easting, northing, threshold, "The selected place is outside the prepared Nelson raster extent.")
            window = Window(column_start, row_start, column_stop - column_start, row_stop - row_start)
            reference_transform = reference.transform
            reference_crs = reference.crs.to_string()
            reference_profile = reference.profile.copy()

        arrays: dict[str, np.ndarray] = {}
        nodata: dict[str, float | int | None] = {}
        for name, path in self.paths.items():
            with rasterio.open(path) as source:
                signature_matches = (
                    source.width == reference_profile["width"]
                    and source.height == reference_profile["height"]
                    and source.transform == reference_transform
                    and source.crs is not None
                    and source.crs.to_string() == reference_crs
                )
                if not signature_matches:
                    raise ValueError(f"Raster alignment mismatch for {path.name}.")
                arrays[name] = source.read(1, window=window)
                nodata[name] = source.nodata

        rows = np.arange(row_start, row_stop, dtype=np.float64) + 0.5
        columns = np.arange(column_start, column_stop, dtype=np.float64) + 0.5
        x_coordinates = reference_transform.c + columns * reference_transform.a
        y_coordinates = reference_transform.f + rows * reference_transform.e
        circle = (
            (x_coordinates[None, :] - easting) ** 2
            + (y_coordinates[:, None] - northing) ** 2
            <= float(radius_metres) ** 2
        )

        score_valid = np.isfinite(arrays["score"])
        if nodata["score"] is not None:
            score_valid &= ~np.isclose(arrays["score"], float(nodata["score"]))
        binary_valid = np.ones_like(circle, dtype=bool)
        if nodata["binary"] is not None:
            binary_valid &= arrays["binary"] != nodata["binary"]
        evaluated = circle & (arrays["coverage"] == 1) & score_valid & binary_valid
        land_in_circle = circle & (arrays["valid_land"] == 1)
        evaluated_cells = int(np.sum(evaluated))
        circle_cells = int(np.sum(circle))
        land_cells = int(np.sum(land_in_circle))
        cell_area_m2 = resolution_x * resolution_y

        if evaluated_cells == 0:
            return self._unavailable(place, radius_metres, easting, northing, threshold, "No V3 model-covered land cells exist inside the selected radius.")

        scores = arrays["score"][evaluated].astype(np.float32)
        flagged = evaluated & (arrays["binary"] == 1)
        flagged_cells = int(np.sum(flagged))
        elevations = arrays["elevation"][evaluated].astype(np.float32)
        slopes = arrays["slope"][evaluated].astype(np.float32)
        distances = arrays["distance"][evaluated].astype(np.float32)
        quantiles = np.quantile(scores, [0.05, 0.25, 0.50, 0.75, 0.95])

        class_counts = Counter(int(value) for value in arrays["land_cover"][flagged])
        leading_classes = [
            {
                "class_code": class_code,
                "cells": count,
                "percentage_of_flagged_cells": 100.0 * count / flagged_cells if flagged_cells else 0.0,
            }
            for class_code, count in class_counts.most_common(5)
        ]
        risk_band_cells = {
            "lower_indicator": int(np.sum(scores < LOWER_REVIEW_BOUND)),
            "review_below_threshold": int(np.sum((scores >= LOWER_REVIEW_BOUND) & (scores < threshold))),
            "operational_susceptibility": int(np.sum((scores >= threshold) & (scores < UPPER_REVIEW_BOUND))),
            "elevated_susceptibility_score": int(np.sum(scores >= UPPER_REVIEW_BOUND)),
        }
        risk_band_percentages = {
            key: 100.0 * value / evaluated_cells for key, value in risk_band_cells.items()
        }
        flagged_percentage = 100.0 * flagged_cells / evaluated_cells
        plain_summary = (
            f"Within {radius_metres:,} metres of {place['display_name']}, "
            f"{flagged_percentage:.2f}% of model-covered land cells meet the frozen "
            f"V3 screening threshold. These locations are recommended for further "
            "human and technical review; the score is not a flood probability."
        )

        run_id = uuid.uuid4().hex[:12]
        run_directory = self.root / "outputs" / "place_analysis_v3" / f"{safe_slug(place['display_name'])}_{radius_metres}m_{run_id}"
        run_directory.mkdir(parents=True, exist_ok=True)
        map_path = run_directory / "v3_local_susceptibility_map.png"
        result_path = run_directory / "v3_place_analysis.json"
        result = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "analysis_ready",
            "analysis_type": "V3 radius-based regional susceptibility screening",
            "place": {**place, "easting_nztm2000": float(easting), "northing_nztm2000": float(northing)},
            "radius_metres": radius_metres,
            "crs": reference_crs,
            "model": "V3 logistic regression",
            "model_tribe": "symbolist/statistical linear classifier",
            "operational_threshold": threshold,
            "threshold_source": summary.get("threshold_source", "V3 validation split"),
            "score_is_calibrated_probability": False,
            "model_retrained": False,
            "threshold_modified": False,
            "flood_target_raster_accessed": False,
            "circle_cells": circle_cells,
            "land_cells_inside_radius": land_cells,
            "evaluated_land_cells": evaluated_cells,
            "evaluated_land_coverage_percentage": 100.0 * evaluated_cells / land_cells if land_cells else 0.0,
            "search_circle_coverage_percentage": 100.0 * evaluated_cells / circle_cells,
            "evaluated_land_area_square_kilometres": evaluated_cells * cell_area_m2 / 1_000_000.0,
            "susceptible_cells": flagged_cells,
            "susceptible_area_square_kilometres": flagged_cells * cell_area_m2 / 1_000_000.0,
            "susceptible_percentage_of_evaluated_land": flagged_percentage,
            "mean_susceptibility_score": float(np.mean(scores)),
            "median_susceptibility_score": float(np.median(scores)),
            "maximum_susceptibility_score": float(np.max(scores)),
            "susceptibility_score_quantiles": {
                "p05": float(quantiles[0]), "p25": float(quantiles[1]),
                "p50": float(quantiles[2]), "p75": float(quantiles[3]),
                "p95": float(quantiles[4]),
            },
            "risk_band_cells": risk_band_cells,
            "risk_band_percentages": risk_band_percentages,
            "median_elevation_metres": float(np.median(elevations)),
            "median_slope_degrees": float(np.median(slopes)),
            "median_distance_to_coast_metres": float(np.median(distances)),
            "leading_susceptible_land_cover_codes": leading_classes,
            "screening_outcome": "Further investigation recommended" if flagged_cells else "No cells flagged at the frozen threshold",
            "plain_language_summary": plain_summary,
            "interpretation_warning": "Screening-level susceptibility only; not a calibrated probability, validated flood extent, engineering design product or emergency-warning service.",
            "artifacts": {
                "analysis_json": str(result_path.relative_to(self.root)),
                "local_map": str(map_path.relative_to(self.root)),
            },
        }
        self._create_map(arrays, evaluated, flagged, window, reference_transform, place, easting, northing, radius_metres, threshold, map_path)
        result_path.write_text(json.dumps(result, indent=2, default=json_safe), encoding="utf-8")
        return result

    def _unavailable(self, place: dict, radius_metres: int, easting: float, northing: float, threshold: float, reason: str) -> dict:
        return {
            "run_id": uuid.uuid4().hex[:12],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "data_unavailable",
            "analysis_type": "V3 radius-based regional susceptibility screening",
            "place": {**place, "easting_nztm2000": float(easting), "northing_nztm2000": float(northing)},
            "radius_metres": radius_metres,
            "model": "V3 logistic regression",
            "operational_threshold": threshold,
            "score_is_calibrated_probability": False,
            "reason": reason,
            "model_retrained": False,
            "threshold_modified": False,
            "flood_target_raster_accessed": False,
            "artifacts": {},
        }

    def _create_map(self, arrays: dict, evaluated: np.ndarray, flagged: np.ndarray, window: Window, transform, place: dict, easting: float, northing: float, radius_metres: int, threshold: float, output_path: Path) -> None:
        left, bottom, right, top = window_bounds(window, transform)
        extent = [left, right, bottom, top]
        elevation = np.ma.masked_where(~evaluated, arrays["elevation"])
        score = np.ma.masked_where(~evaluated, arrays["score"])
        binary = np.ma.masked_where(~evaluated, arrays["binary"])
        distance = np.ma.masked_where(~evaluated, arrays["distance"])
        valid_scores = arrays["score"][evaluated]
        score_vmax = max(float(np.quantile(valid_scores, 0.99)), threshold * 2.0, UPPER_REVIEW_BOUND)
        score_vmax = min(score_vmax, 1.0)

        figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
        figure.suptitle(f"ClimateAdapt AI V3 – {place['display_name']} ({radius_metres:,} m radius)", fontsize=17, fontweight="bold")
        image = axes[0, 0].imshow(elevation, extent=extent, origin="upper", cmap="terrain")
        axes[0, 0].set_title("Elevation input")
        figure.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Metres")

        image = axes[0, 1].imshow(score, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=score_vmax)
        axes[0, 1].set_title("V3 susceptibility score (display-scaled)")
        figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Uncalibrated score")

        axes[1, 0].imshow(binary, extent=extent, origin="upper", cmap=ListedColormap(["#eeeeee", "#d62728"]), vmin=0, vmax=1)
        axes[1, 0].set_title(f"Flagged land (frozen threshold {threshold:.4f})")

        image = axes[1, 1].imshow(distance, extent=extent, origin="upper", cmap="cividis")
        axes[1, 1].set_title("Distance to coastline")
        figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Metres")

        for axis in axes.flat:
            axis.scatter([easting], [northing], marker="+", s=80, color="black")
            axis.set_xlabel("Easting (NZTM2000)")
            axis.set_ylabel("Northing (NZTM2000)")
            axis.ticklabel_format(style="plain", useOffset=False)
        figure.text(0.5, 0.005, "Frozen V3 regional screening; scores are not probabilities; human review required.", ha="center", fontsize=10)
        figure.savefig(output_path, dpi=170, bbox_inches="tight")
        plt.close(figure)
