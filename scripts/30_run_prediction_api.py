"""Serve instant ClimateAdapt V3 point screening from precomputed rasters.

Run from the ClimateAdapt_AI_Starter project root:

    python .\\scripts\\30_run_prediction_api.py

Then open http://127.0.0.1:8000/docs for the interactive API documentation.

This service does not rerun a model. It reads the frozen V3 logistic regional
GeoTIFF products at a requested coordinate, making queries fast and independent
of GPU availability. Existing V2 U-Net rasters are returned only as an optional
comparison when they are available and aligned.
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
OUTPUT_DIR = ROOT / "outputs" / "full_region_inference_v3"
V2_OUTPUT_DIR = ROOT / "outputs" / "full_region_inference_v2"
SOURCE_RASTER_DIR = ROOT / "data" / "processed" / "full_region" / "rasters"

RASTER_PATHS = {
    "susceptibility_score": OUTPUT_DIR / "nelson_logistic_susceptibility_score_v3.tif",
    "susceptibility": OUTPUT_DIR / "nelson_logistic_susceptibility_v3.tif",
    "coverage": OUTPUT_DIR / "nelson_logistic_coverage_v3.tif",
}

OPTIONAL_V2_COMPARISON_PATHS = {
    "v2_probability": V2_OUTPUT_DIR / "nelson_probability_v2.tif",
    "v2_susceptibility": V2_OUTPUT_DIR / "nelson_susceptibility_v2.tif",
    "v2_uncertainty": V2_OUTPUT_DIR / "nelson_uncertainty_v2.tif",
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

SUMMARY_PATH = OUTPUT_DIR / "nelson_logistic_full_region_summary_v3.json"
TEST_SUMMARY_PATH = ROOT / "outputs" / "test_v3" / "v3_locked_test_summary.json"
DEFAULT_THRESHOLD = 0.004
V2_THRESHOLD = 0.54
DEFAULT_UNCERTAINTY_REVIEW_THRESHOLD = 0.80
NEAR_THRESHOLD_LOWER = 0.002
NEAR_THRESHOLD_UPPER = 0.010


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
        self.test_summary: dict[str, Any] = {}
        self.reference_name = "susceptibility_score"
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

        for name, path in OPTIONAL_V2_COMPARISON_PATHS.items():
            if path.exists():
                candidate = rasterio.open(path)
                if self._is_aligned(candidate, self.reference):
                    self.datasets[name] = candidate
                else:
                    # An optional historical comparison must never prevent the
                    # primary V3 service from starting.
                    candidate.close()

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
        if TEST_SUMMARY_PATH.exists():
            with TEST_SUMMARY_PATH.open("r", encoding="utf-8") as source:
                self.test_summary = json.load(source)

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
            if not self._is_aligned(dataset, reference):
                mismatches.append(name)
        if mismatches:
            raise RuntimeError(
                "Raster alignment check failed for: " + ", ".join(sorted(mismatches))
            )

    @staticmethod
    def _is_aligned(
        dataset: rasterio.io.DatasetReader,
        reference: rasterio.io.DatasetReader,
    ) -> bool:
        return (
            dataset.crs == reference.crs
            and dataset.transform == reference.transform
            and dataset.width == reference.width
            and dataset.height == reference.height
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

        if name in {
            "susceptibility",
            "coverage",
            "land_cover_class",
            "valid_land_mask",
            "v2_susceptibility",
        }:
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
    version="3.0.0",
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
    pooled_metrics = (
        STORE.test_summary.get("logistic", {}).get("pooled_metrics", {})
        if isinstance(STORE.test_summary, dict)
        else {}
    )
    return {
        "experiment": summary.get(
            "experiment", "ClimateAdapt V3 frozen logistic full-region inference"
        ),
        "primary_model": "V3 logistic regression",
        "model_tribe": summary.get(
            "model_tribe", "symbolist/statistical linear classifier"
        ),
        "model_path": summary.get(
            "model_path", "models/v3/logistic_regression_v3.joblib"
        ),
        "operational_threshold": summary.get("frozen_threshold", DEFAULT_THRESHOLD),
        "threshold_source": summary.get(
            "threshold_source", "V3 Boulder Bank and Delaware Bay validation split"
        ),
        "input_channels": summary.get("input_channels", 11),
        "score_is_calibrated_probability": False,
        "locked_test_metrics": {
            key: pooled_metrics.get(key)
            for key in (
                "average_precision",
                "precision",
                "recall",
                "f2_score",
                "dice",
                "balanced_accuracy",
            )
            if key in pooled_metrics
        },
        "optional_v2_comparison_available": "v2_probability" in STORE.datasets,
        "serving_method": "precomputed aligned GeoTIFF lookup",
        "flood_labels_accessed_during_serving": False,
        "warning": (
            "Screening-level susceptibility only; not a validated flood extent, "
            "engineering design product, or emergency-warning service."
        ),
    }


def risk_band(score: float, threshold: float) -> str:
    """Operational score bands; these are not probability categories."""
    lower_review = min(NEAR_THRESHOLD_LOWER, threshold)
    upper_review = max(NEAR_THRESHOLD_UPPER, threshold)
    if score < lower_review:
        return "lower_indicator"
    if score < threshold:
        return "review_below_threshold"
    if score < upper_review:
        return "operational_susceptibility"
    return "elevated_susceptibility_score"


@app.post("/predict/point")
def predict_point(request: PointPredictionRequest) -> dict[str, Any]:
    easting, northing, latitude, longitude = STORE.coordinate_pair(request)
    row, column = STORE.pixel_location(easting, northing)

    coverage = STORE.read_value("coverage", row, column)
    score = STORE.read_value("susceptibility_score", row, column)
    susceptibility = STORE.read_value("susceptibility", row, column)
    valid_land = STORE.read_value("valid_land_mask", row, column)

    if coverage in {None, 0} or score is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "The coordinate is inside the regional grid but has no valid model "
                "coverage, usually because it is offshore or lacks DEM coverage."
            ),
        )
    if valid_land == 0:
        raise HTTPException(status_code=404, detail="The requested coordinate is not valid land.")

    score = float(score)
    threshold = float(STORE.summary.get("frozen_threshold", DEFAULT_THRESHOLD))
    binary_value = (
        bool(int(susceptibility))
        if susceptibility is not None
        else score >= threshold
    )

    terrain = {
        "elevation_metres": STORE.read_value("elevation_metres", row, column),
        "slope_degrees": STORE.read_value("slope_degrees", row, column),
        "distance_to_coast_metres": STORE.read_value(
            "distance_to_coast_metres", row, column
        ),
        "land_cover_class": STORE.read_value("land_cover_class", row, column),
    }

    band = risk_band(score, threshold)
    review_reasons: list[str] = [
        "all planning interpretations require authorised human review"
    ]

    if binary_value:
        review_reasons.append(
            "the point meets the frozen operational susceptibility threshold"
        )

    if NEAR_THRESHOLD_LOWER <= score < NEAR_THRESHOLD_UPPER:
        review_reasons.append(
            "susceptibility score is in the predeclared near-threshold review band"
        )

    comparison = None
    v2_probability = STORE.read_value("v2_probability", row, column)
    if v2_probability is not None:
        v2_binary = STORE.read_value("v2_susceptibility", row, column)
        v2_uncertainty = STORE.read_value("v2_uncertainty", row, column)
        comparison = {
            "model": "GroupNorm Compact U-Net V2",
            "probability": float(v2_probability),
            "operational_threshold": V2_THRESHOLD,
            "susceptible": (
                bool(int(v2_binary))
                if v2_binary is not None
                else float(v2_probability) >= V2_THRESHOLD
            ),
            "normalised_predictive_uncertainty": (
                float(v2_uncertainty) if v2_uncertainty is not None else None
            ),
            "role": "historical optional comparison; not the primary V3 result",
        }

    return {
        "status": "human_review_required" if review_reasons else "screening_result_ready",
        "location": {
            "name": request.location_name,
            "nztm2000": {"easting": easting, "northing": northing},
            "wgs84": {"latitude": latitude, "longitude": longitude},
            "grid": {"row": row, "column": column},
        },
        "prediction": {
            "susceptibility_score": score,
            "score_is_calibrated_probability": False,
            # Temporary compatibility alias for the existing V1 agent/dashboard.
            # Downstream V3 code should read susceptibility_score instead.
            "probability": score,
            "probability_field_deprecated": True,
            "operational_threshold": threshold,
            "susceptible": binary_value,
            "risk_band": band,
            "normalised_predictive_uncertainty": None,
        },
        "comparison": comparison,
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
            "model": "V3 logistic regression",
            "model_path": STORE.summary.get(
                "model_path", "models/v3/logistic_regression_v3.joblib"
            ),
            "model_tribe": "symbolist/statistical linear classifier",
            "threshold_source": STORE.summary.get(
                "threshold_source",
                "V3 Boulder Bank and Delaware Bay validation split",
            ),
            "serving_method": "precomputed regional raster lookup",
            "model_retrained": False,
            "threshold_modified": False,
            "flood_labels_accessed_during_serving": False,
        },
        "limitations": [
            "Planning-level susceptibility screening only.",
            "The susceptibility score is not a literal or calibrated flood probability.",
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
