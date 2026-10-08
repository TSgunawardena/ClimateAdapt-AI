"""Serve instant ClimateAdapt point predictions from precomputed Nelson rasters.

Run from the ClimateAdapt_AI_Starter project root:

    python .\\scripts\\30_run_prediction_api.py

Then open http://127.0.0.1:8000/docs for the interactive API documentation.

This service does not rerun the U-Net. It reads the already generated regional
GeoTIFF products at a requested coordinate, making point queries fast and
independent of GPU availability.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyproj import CRS, Transformer
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "full_region_inference_v2"
SOURCE_RASTER_DIR = ROOT / "data" / "processed" / "full_region" / "rasters"

RASTER_PATHS = {
    "probability": OUTPUT_DIR / "nelson_probability_v2.tif",
    "susceptibility": OUTPUT_DIR / "nelson_susceptibility_v2.tif",
    "uncertainty": OUTPUT_DIR / "nelson_uncertainty_v2.tif",
    "coverage": OUTPUT_DIR / "nelson_inference_coverage_v2.tif",
}

OPTIONAL_RASTER_CANDIDATES = {
    "elevation_metres": ["elevation_5m.tif", "elevation.tif"],
    "slope_degrees": ["slope_degrees_5m.tif", "slope_5m.tif"],
    "distance_to_coast_metres": [
        "distance_to_coast_5m.tif",
        "distance_to_coastline_5m.tif",
    ],
    "land_cover_class": ["land_cover_class_5m.tif", "land_cover_5m.tif"],
    "valid_land_mask": ["valid_land_mask_5m.tif", "land_mask_5m.tif"],
}

SUMMARY_PATH = OUTPUT_DIR / "nelson_full_region_summary_v2.json"
DEFAULT_THRESHOLD = 0.54
DEFAULT_UNCERTAINTY_REVIEW_THRESHOLD = 0.80


class PointPredictionRequest(BaseModel):
    """One location expressed in either NZTM2000 or WGS84 coordinates."""

    easting: float | None = Field(
        default=None,
        description="NZTM2000 easting (EPSG:2193). Supply with northing.",
    )
    northing: float | None = Field(
        default=None,
        description="NZTM2000 northing (EPSG:2193). Supply with easting.",
    )
    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        description="WGS84 latitude. Supply with longitude.",
    )
    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        description="WGS84 longitude. Supply with latitude.",
    )
    location_name: str | None = Field(
        default=None,
        max_length=120,
        description="Optional user-facing label; it is not used for geocoding.",
    )


class RasterStore:
    """Own aligned raster datasets and provide safe single-pixel reads."""

    def __init__(self) -> None:
        self.datasets: dict[str, rasterio.io.DatasetReader] = {}
        self.summary: dict[str, Any] = {}
        self.reference_name = "probability"
        self.to_wgs84: Transformer | None = None
        self.from_wgs84: Transformer | None = None

    @property
    def reference(self) -> rasterio.io.DatasetReader:
        return self.datasets[self.reference_name]

    def open(self) -> None:
        missing = [str(path.relative_to(ROOT)) for path in RASTER_PATHS.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Required full-region inference rasters are missing: " + ", ".join(missing)
            )

        for name, path in RASTER_PATHS.items():
            self.datasets[name] = rasterio.open(path)

        for output_name, candidates in OPTIONAL_RASTER_CANDIDATES.items():
            for filename in candidates:
                path = SOURCE_RASTER_DIR / filename
                if path.exists():
                    self.datasets[output_name] = rasterio.open(path)
                    break

        self._validate_alignment()

        if SUMMARY_PATH.exists():
            with SUMMARY_PATH.open("r", encoding="utf-8") as source:
                self.summary = json.load(source)

        raster_crs = CRS.from_user_input(self.reference.crs)
        self.to_wgs84 = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        self.from_wgs84 = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

    def close(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()

    def _validate_alignment(self) -> None:
        reference = self.reference
        mismatches: list[str] = []
        for name, dataset in self.datasets.items():
            if (
                dataset.crs != reference.crs
                or dataset.transform != reference.transform
                or dataset.width != reference.width
                or dataset.height != reference.height
            ):
                mismatches.append(name)
        if mismatches:
            raise RuntimeError(
                "Raster alignment check failed for: " + ", ".join(sorted(mismatches))
            )

    def coordinate_pair(self, request: PointPredictionRequest) -> tuple[float, float, float, float]:
        has_nztm_any = request.easting is not None or request.northing is not None
        has_wgs_any = request.latitude is not None or request.longitude is not None
        has_nztm_pair = request.easting is not None and request.northing is not None
        has_wgs_pair = request.latitude is not None and request.longitude is not None

        if has_nztm_any and not has_nztm_pair:
            raise HTTPException(status_code=422, detail="Supply both easting and northing.")
        if has_wgs_any and not has_wgs_pair:
            raise HTTPException(status_code=422, detail="Supply both latitude and longitude.")
        if has_nztm_pair == has_wgs_pair:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Supply exactly one complete coordinate pair: either "
                    "easting/northing or latitude/longitude."
                ),
            )

        if has_nztm_pair:
            easting = float(request.easting)
            northing = float(request.northing)
            assert self.to_wgs84 is not None
            longitude, latitude = self.to_wgs84.transform(easting, northing)
        else:
            latitude = float(request.latitude)
            longitude = float(request.longitude)
            assert self.from_wgs84 is not None
            easting, northing = self.from_wgs84.transform(longitude, latitude)

        return easting, northing, latitude, longitude

    def pixel_location(self, easting: float, northing: float) -> tuple[int, int]:
        bounds = self.reference.bounds
        if not (bounds.left <= easting < bounds.right and bounds.bottom < northing <= bounds.top):
            raise HTTPException(
                status_code=404,
                detail="The requested coordinate is outside the prepared Nelson raster extent.",
            )

        row, column = self.reference.index(easting, northing)
        if not (0 <= row < self.reference.height and 0 <= column < self.reference.width):
            raise HTTPException(status_code=404, detail="The requested coordinate is outside the raster grid.")
        return int(row), int(column)

    def read_value(self, name: str, row: int, column: int) -> float | int | None:
        dataset = self.datasets.get(name)
        if dataset is None:
            return None

        value = dataset.read(1, window=Window(column, row, 1, 1))[0, 0]
        if np.ma.is_masked(value):
            return None
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        if dataset.nodata is not None and np.isclose(numeric, float(dataset.nodata)):
            return None

        if name in {"susceptibility", "coverage", "land_cover_class", "valid_land_mask"}:
            return int(round(numeric))
        return numeric


STORE = RasterStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    STORE.open()
    try:
        yield
    finally:
        STORE.close()


app = FastAPI(
    title="ClimateAdapt AI Prediction API",
    version="1.0.0",
    description=(
        "Instant planning-level coastal flood susceptibility queries from the "
        "precomputed Nelson regional raster products."
    ),
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "ClimateAdapt AI Prediction API",
        "status": "ready",
        "documentation": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict/point",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "rasters_open": sorted(STORE.datasets),
        "crs": str(STORE.reference.crs),
        "grid": {
            "width": STORE.reference.width,
            "height": STORE.reference.height,
            "resolution_metres": abs(float(STORE.reference.transform.a)),
        },
        "gpu_required": False,
    }


@app.get("/model")
def model_information() -> dict[str, Any]:
    summary = STORE.summary
    return {
        "experiment": summary.get("experiment", "Experiment 2 – GroupNorm Compact U-Net"),
        "checkpoint": summary.get("checkpoint", "models/climateadapt_unet_v2_best.pt"),
        "checkpoint_epoch": summary.get("checkpoint_epoch", 4),
        "operational_threshold": summary.get("operational_threshold", DEFAULT_THRESHOLD),
        "threshold_source": summary.get("threshold_source", "Rocks Road validation region"),
        "input_channels": summary.get("input_channels", 25),
        "serving_method": "precomputed aligned GeoTIFF lookup",
        "flood_labels_accessed_during_serving": False,
        "warning": (
            "Screening-level susceptibility only; not a validated flood extent, "
            "engineering design product, or emergency-warning service."
        ),
    }


def risk_band(probability: float, threshold: float) -> str:
    if probability < 0.35:
        return "lower_indicator"
    if probability < threshold:
        return "review_band"
    if probability < 0.75:
        return "operational_susceptibility"
    return "high_indicator"


@app.post("/predict/point")
def predict_point(request: PointPredictionRequest) -> dict[str, Any]:
    easting, northing, latitude, longitude = STORE.coordinate_pair(request)
    row, column = STORE.pixel_location(easting, northing)

    coverage = STORE.read_value("coverage", row, column)
    probability = STORE.read_value("probability", row, column)
    susceptibility = STORE.read_value("susceptibility", row, column)
    uncertainty = STORE.read_value("uncertainty", row, column)
    valid_land = STORE.read_value("valid_land_mask", row, column)

    if coverage in {None, 0} or probability is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "The coordinate is inside the regional grid but has no valid model "
                "coverage, usually because it is offshore or lacks DEM coverage."
            ),
        )
    if valid_land == 0:
        raise HTTPException(status_code=404, detail="The requested coordinate is not valid land.")

    probability = float(probability)
    uncertainty_value = float(uncertainty) if uncertainty is not None else None
    threshold = float(STORE.summary.get("operational_threshold", DEFAULT_THRESHOLD))
    binary_value = bool(int(susceptibility)) if susceptibility is not None else probability >= threshold

    terrain = {
        "elevation_metres": STORE.read_value("elevation_metres", row, column),
        "slope_degrees": STORE.read_value("slope_degrees", row, column),
        "distance_to_coast_metres": STORE.read_value(
            "distance_to_coast_metres", row, column
        ),
        "land_cover_class": STORE.read_value("land_cover_class", row, column),
    }

    band = risk_band(probability, threshold)
    review_reasons: list[str] = []
    if band == "review_band":
        review_reasons.append("probability is close to the operational threshold")
    if uncertainty_value is not None and uncertainty_value >= DEFAULT_UNCERTAINTY_REVIEW_THRESHOLD:
        review_reasons.append("normalised predictive uncertainty is high")

    return {
        "status": "human_review_required" if review_reasons else "screening_result_ready",
        "location": {
            "name": request.location_name,
            "nztm2000": {"easting": easting, "northing": northing},
            "wgs84": {"latitude": latitude, "longitude": longitude},
            "grid": {"row": row, "column": column},
        },
        "prediction": {
            "probability": probability,
            "operational_threshold": threshold,
            "susceptible": binary_value,
            "risk_band": band,
            "normalised_predictive_uncertainty": uncertainty_value,
        },
        "terrain_context": terrain,
        "coverage": {
            "valid_model_coverage": True,
            "resolution_metres": abs(float(STORE.reference.transform.a)),
            "crs": str(STORE.reference.crs),
        },
        "human_review": {
            "required": bool(review_reasons),
            "reasons": review_reasons,
        },
        "provenance": {
            "model": "GroupNorm Compact U-Net V2",
            "checkpoint_epoch": STORE.summary.get("checkpoint_epoch", 4),
            "threshold_source": STORE.summary.get(
                "threshold_source", "Rocks Road validation region"
            ),
            "serving_method": "precomputed regional raster lookup",
            "model_retrained": False,
            "threshold_modified": False,
        },
        "limitations": [
            "Planning-level susceptibility screening only.",
            "Not a hydrodynamic flood-depth or flood-velocity simulation.",
            "Not an emergency-warning, regulatory, or engineering-design product.",
            "GIS, engineering, community and human review remain required.",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("CLIMATEADAPT_API_HOST", "127.0.0.1")
    port = int(os.getenv("CLIMATEADAPT_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
