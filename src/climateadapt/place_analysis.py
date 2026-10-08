"""Windowed place analysis from the precomputed full-region V2 rasters."""

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


BUILTIN_PLACES = {
    "atawhai": {
        "name": "Atawhai, Nelson",
        "latitude": -41.2479,
        "longitude": 173.3277,
    },
    "nelson": {
        "name": "Nelson CBD",
        "latitude": -41.2746,
        "longitude": 173.2839,
    },
    "nelson cbd": {
        "name": "Nelson CBD",
        "latitude": -41.2746,
        "longitude": 173.2839,
    },
    "tahunanui": {
        "name": "Tāhunanui, Nelson",
        "latitude": -41.2844,
        "longitude": 173.2450,
    },
    "tāhunanui": {
        "name": "Tāhunanui, Nelson",
        "latitude": -41.2844,
        "longitude": 173.2450,
    },
    "stoke": {
        "name": "Stoke, Nelson",
        "latitude": -41.3120,
        "longitude": 173.2320,
    },
    "monaco": {
        "name": "Monaco, Nelson",
        "latitude": -41.3000,
        "longitude": 173.2185,
    },
}


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:60] or "selected-place"


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


class PlaceAnalysisService:
    """Resolve a place and summarise existing regional model outputs."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        self.raster_dir = (
            self.root / "data" / "processed" / "full_region" / "rasters"
        )
        self.inference_dir = self.root / "outputs" / "full_region_inference_v2"
        self.threshold_path = (
            self.root
            / "outputs"
            / "validation_v2"
            / "selected_threshold_v2.json"
        )
        self.paths = {
            "probability": self.inference_dir / "nelson_probability_v2.tif",
            "binary": self.inference_dir / "nelson_susceptibility_v2.tif",
            "uncertainty": self.inference_dir / "nelson_uncertainty_v2.tif",
            "coverage": self.inference_dir / "nelson_inference_coverage_v2.tif",
            "elevation": self.raster_dir / "elevation_5m.tif",
            "slope": self.raster_dir / "slope_degrees_5m.tif",
            "distance": self.raster_dir / "distance_to_coast_5m.tif",
            "land_cover": self.raster_dir / "land_cover_class_5m.tif",
        }

    def required_files(self) -> list[Path]:
        return [*self.paths.values(), self.threshold_path]

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

        normalised = re.sub(r"\s+", " ", query.strip().lower())
        if not normalised:
            raise ValueError("Enter a place name or provide manual coordinates.")
        if normalised in BUILTIN_PLACES:
            place = BUILTIN_PLACES[normalised]
            return {
                "query": query.strip(),
                "display_name": place["name"],
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "source": "offline_nelson_catalogue",
            }

        search_query = query.strip()
        if "new zealand" not in normalised and "nz" not in normalised:
            search_query = f"{search_query}, Nelson, New Zealand"
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": search_query,
                    "format": "jsonv2",
                    "limit": 5,
                    "countrycodes": "nz",
                },
                headers={
                    "User-Agent": "ClimateAdaptAI-NMIT-Academic-Prototype/1.0"
                },
                timeout=10,
            )
            response.raise_for_status()
            candidates = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(
                "Online place search was unavailable. Select a built-in Nelson "
                "place or use manual latitude/longitude."
            ) from error

        if not candidates:
            raise ValueError(
                f"No matching New Zealand location was found for '{query.strip()}'."
            )

        candidate = candidates[0]
        return {
            "query": query.strip(),
            "display_name": str(candidate.get("display_name", query.strip())),
            "latitude": float(candidate["lat"]),
            "longitude": float(candidate["lon"]),
            "source": "OpenStreetMap Nominatim",
        }

    def analyse(self, place: dict, radius_metres: int) -> dict:
        if radius_metres not in {500, 1000, 2000}:
            raise ValueError("Radius must be 500, 1000 or 2000 metres.")
        missing = self.missing_files()
        if missing:
            raise FileNotFoundError(
                "Missing required regional outputs: "
                + ", ".join(str(path.relative_to(self.root)) for path in missing)
            )

        with self.threshold_path.open("r", encoding="utf-8") as file:
            threshold_record = json.load(file)
        threshold = float(
            threshold_record.get(
                "operational_threshold", threshold_record.get("selected_threshold")
            )
        )

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
        easting, northing = transformer.transform(
            float(place["longitude"]), float(place["latitude"])
        )

        with rasterio.open(self.paths["probability"]) as reference:
            if reference.crs is None or reference.crs.to_string() != "EPSG:2193":
                raise ValueError("Expected full-region outputs in EPSG:2193.")
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
                return self._unavailable_result(
                    place, radius_metres, easting, northing, threshold,
                    "The selected point is outside the regional raster extent.",
                )
            window = Window(
                column_start,
                row_start,
                column_stop - column_start,
                row_stop - row_start,
            )
            reference_transform = reference.transform
            reference_crs = reference.crs.to_string()
            reference_profile = reference.profile.copy()

        arrays = {}
        nodata_values = {}
        for name, path in self.paths.items():
            with rasterio.open(path) as source:
                if (
                    source.width != reference_profile["width"]
                    or source.height != reference_profile["height"]
                    or source.transform != reference_transform
                    or source.crs.to_string() != reference_crs
                ):
                    raise ValueError(f"Raster alignment mismatch for {path.name}.")
                arrays[name] = source.read(1, window=window)
                nodata_values[name] = source.nodata

        row_indexes = np.arange(row_start, row_stop, dtype=np.float64) + 0.5
        column_indexes = np.arange(column_start, column_stop, dtype=np.float64) + 0.5
        x_coordinates = (
            reference_transform.c + column_indexes * reference_transform.a
        )
        y_coordinates = reference_transform.f + row_indexes * reference_transform.e
        circle = (
            (x_coordinates[None, :] - easting) ** 2
            + (y_coordinates[:, None] - northing) ** 2
            <= float(radius_metres) ** 2
        )

        probability_valid = arrays["probability"] != nodata_values["probability"]
        binary_valid = arrays["binary"] != nodata_values["binary"]
        uncertainty_valid = arrays["uncertainty"] != nodata_values["uncertainty"]
        evaluated = (
            circle
            & (arrays["coverage"] == 1)
            & probability_valid
            & binary_valid
            & uncertainty_valid
        )
        circle_cells = int(np.sum(circle))
        evaluated_cells = int(np.sum(evaluated))
        cell_area_m2 = resolution_x * resolution_y

        if evaluated_cells == 0:
            return self._unavailable_result(
                place, radius_metres, easting, northing, threshold,
                "No evaluated land cells exist inside the selected radius.",
            )

        probabilities = arrays["probability"][evaluated].astype(np.float32)
        uncertainties = arrays["uncertainty"][evaluated].astype(np.float32)
        susceptible = evaluated & (arrays["binary"] == 1)
        susceptible_cells = int(np.sum(susceptible))
        elevations = arrays["elevation"][evaluated].astype(np.float32)
        slopes = arrays["slope"][evaluated].astype(np.float32)
        distances = arrays["distance"][evaluated].astype(np.float32)

        class_counts = Counter(int(value) for value in arrays["land_cover"][susceptible])
        leading_classes = [
            {
                "class_code": class_code,
                "cells": count,
                "percentage_of_susceptible_cells": (
                    100.0 * count / susceptible_cells if susceptible_cells else 0.0
                ),
            }
            for class_code, count in class_counts.most_common(5)
        ]

        run_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()
        run_directory = (
            self.root
            / "outputs"
            / "place_analysis_v1"
            / f"{safe_slug(place['display_name'])}_{radius_metres}m_{run_id}"
        )
        run_directory.mkdir(parents=True, exist_ok=True)
        map_path = run_directory / "local_susceptibility_map.png"
        result_path = run_directory / "place_analysis.json"

        result = {
            "run_id": run_id,
            "created_at_utc": timestamp,
            "status": "analysis_ready",
            "place": {
                **place,
                "easting_nztm2000": float(easting),
                "northing_nztm2000": float(northing),
            },
            "radius_metres": radius_metres,
            "crs": reference_crs,
            "operational_threshold": threshold,
            "threshold_source": threshold_record.get("selection_dataset"),
            "model_retrained": False,
            "threshold_modified": False,
            "flood_target_raster_accessed": False,
            "circle_cells": circle_cells,
            "evaluated_land_cells": evaluated_cells,
            "search_circle_coverage_percentage": 100.0 * evaluated_cells / circle_cells,
            "evaluated_land_area_square_kilometres": (
                evaluated_cells * cell_area_m2 / 1_000_000.0
            ),
            "susceptible_cells": susceptible_cells,
            "susceptible_area_square_kilometres": (
                susceptible_cells * cell_area_m2 / 1_000_000.0
            ),
            "susceptible_percentage_of_evaluated_land": (
                100.0 * susceptible_cells / evaluated_cells
            ),
            "mean_probability": float(np.mean(probabilities)),
            "maximum_probability": float(np.max(probabilities)),
            "median_probability": float(np.median(probabilities)),
            "mean_uncertainty": float(np.mean(uncertainties)),
            "high_uncertainty_cells_entropy_0_80_or_above": int(
                np.sum(uncertainties >= 0.80)
            ),
            "high_uncertainty_percentage": float(
                100.0 * np.sum(uncertainties >= 0.80) / evaluated_cells
            ),
            "median_elevation_metres": float(np.median(elevations)),
            "median_slope_degrees": float(np.median(slopes)),
            "median_distance_to_coast_metres": float(np.median(distances)),
            "risk_band_cells": {
                "lower_indicator_below_0_35": int(np.sum(probabilities < 0.35)),
                "review_band_0_35_to_threshold": int(
                    np.sum((probabilities >= 0.35) & (probabilities < threshold))
                ),
                "operational_susceptibility": int(np.sum(probabilities >= threshold)),
                "high_indicator_0_75_or_above": int(np.sum(probabilities >= 0.75)),
            },
            "leading_susceptible_land_cover_codes": leading_classes,
            "interpretation_warning": (
                "Screening-level susceptibility only; not a validated flood extent, "
                "engineering design product or emergency-warning service."
            ),
            "artifacts": {
                "analysis_json": str(result_path.relative_to(self.root)),
                "local_map": str(map_path.relative_to(self.root)),
            },
        }

        self._create_map(
            arrays,
            evaluated,
            circle,
            susceptible,
            window,
            reference_transform,
            place,
            radius_metres,
            threshold,
            map_path,
        )
        result_path.write_text(
            json.dumps(result, indent=2, default=json_safe), encoding="utf-8"
        )
        return result

    def _unavailable_result(
        self,
        place: dict,
        radius_metres: int,
        easting: float,
        northing: float,
        threshold: float,
        reason: str,
    ) -> dict:
        return {
            "run_id": uuid.uuid4().hex[:12],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "data_unavailable",
            "place": {
                **place,
                "easting_nztm2000": float(easting),
                "northing_nztm2000": float(northing),
            },
            "radius_metres": radius_metres,
            "operational_threshold": threshold,
            "reason": reason,
            "model_retrained": False,
            "threshold_modified": False,
            "flood_target_raster_accessed": False,
            "artifacts": {},
        }

    def _create_map(
        self,
        arrays: dict,
        evaluated: np.ndarray,
        circle: np.ndarray,
        susceptible: np.ndarray,
        window: Window,
        transform,
        place: dict,
        radius_metres: int,
        threshold: float,
        output_path: Path,
    ) -> None:
        left, bottom, right, top = window_bounds(window, transform)
        extent = [left, right, bottom, top]
        elevation = np.ma.masked_where(~evaluated, arrays["elevation"])
        probability = np.ma.masked_where(~evaluated, arrays["probability"])
        binary = np.ma.masked_where(~evaluated, arrays["binary"])
        uncertainty = np.ma.masked_where(~evaluated, arrays["uncertainty"])

        figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
        figure.suptitle(
            f"ClimateAdapt AI – {place['display_name']} ({radius_metres:,} m radius)",
            fontsize=17,
            fontweight="bold",
        )

        image = axes[0, 0].imshow(elevation, extent=extent, origin="upper", cmap="terrain")
        axes[0, 0].set_title("Elevation input")
        figure.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Metres")

        image = axes[0, 1].imshow(
            probability, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1
        )
        axes[0, 1].set_title("Susceptibility probability")
        figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Probability")

        axes[1, 0].imshow(
            binary,
            extent=extent,
            origin="upper",
            cmap=ListedColormap(["#eeeeee", "#d62728"]),
            vmin=0,
            vmax=1,
        )
        axes[1, 0].set_title(f"Operational susceptibility (threshold {threshold:.2f})")

        image = axes[1, 1].imshow(
            uncertainty,
            extent=extent,
            origin="upper",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        axes[1, 1].set_title("Normalised entropy uncertainty")
        figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Uncertainty")

        for axis in axes.flat:
            axis.set_xlabel("Easting (NZTM2000)")
            axis.set_ylabel("Northing (NZTM2000)")
            axis.ticklabel_format(style="plain", useOffset=False)

        figure.text(
            0.5,
            0.005,
            "Precomputed regional inference; planning screening only; human review required.",
            ha="center",
            fontsize=10,
        )
        figure.savefig(output_path, dpi=170, bbox_inches="tight")
        plt.close(figure)
