
from __future__ import annotations

from collections import OrderedDict
import json
import logging
import math
import os
from pathlib import Path
import sys
from threading import Lock
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import requests

LOG = logging.getLogger("climateadapt.mobile")


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=200)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a place name.")
        return value.strip()


class PlaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=300)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    @field_validator("name")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A place name is required.")
        return value.strip()


class AreaRequest(PlaceRequest):
    radius_metres: Literal[500, 1000, 2000] = 1000


def create_app(service, catalogue: dict, normalise, root: Path,
               api_url: str = "http://127.0.0.1:8000", online: bool = False,
               http=requests, clock=time.monotonic) -> FastAPI:
    """Dependencies are explicit so tests never need the real model/raster bundle."""
    app = FastAPI(title="ClimateAdapt Mobile Places", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    area_lock, search_lock = Lock(), Lock()
    search_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
    maps: OrderedDict[str, Path] = OrderedDict()
    last_search = float("-inf")

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health():
        missing = service.missing_files()
        return {"ok": True, "mode": "live_places", "contract": "places-v1",
                "area_ready": not missing, "missing_runtime_files": len(missing),
                "online_search": online, "photo_analysis": "not_connected"}

    @app.get("/api/places")
    def places():
        unique = {name: {"display_name": name, "latitude": lat, "longitude": lon,
                         "source": "offline_nelson_catalogue"}
                  for name, lat, lon in catalogue.values()}
        return {"places": list(unique.values()), "online_search": online}

    @app.post("/api/places/resolve")
    def resolve(body: ResolveRequest):
        nonlocal last_search
        key = normalise(body.query)
        if key in catalogue:
            return {"place": service.resolve_place(body.query)}
        if not online:
            raise HTTPException(422, "Online address search is off. Choose a listed Nelson place or enter coordinates.")
        # User-triggered searches only. One process, one upstream search at a time,
        # at most one per second; cache results for one day. No autocomplete.
        if not search_lock.acquire(blocking=False):
            raise HTTPException(429, "A place search is already running. Try again shortly.", headers={"Retry-After": "2"})
        try:
            cached = search_cache.get(key)
            now = clock()
            if cached and now - cached[0] < 86400:
                search_cache.move_to_end(key)
                return {"place": cached[1]}
            if now - last_search < 1.1:
                raise HTTPException(429, "Please wait before another address search.", headers={"Retry-After": "2"})
            last_search = now
            place = service.resolve_place(body.query)
            search_cache[key] = (clock(), place)
            search_cache.move_to_end(key)
            while len(search_cache) > 128:
                search_cache.popitem(last=False)
            return {"place": place}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, "Address search is unavailable. Choose a listed Nelson place or enter coordinates.") from error
        finally:
            search_lock.release()

    @app.post("/api/places/point")
    def point(body: PlaceRequest):
        try:
            # Verify service identity rather than accepting a score from any API.
            model_response = http.get(f"{api_url}/model", timeout=5)
            model_response.raise_for_status()
            model = model_response.json()
            if model.get("primary_model") != "V3 logistic regression" or model.get("score_is_calibrated_probability") is not False:
                raise ValueError("Unexpected prediction model.")
            response = http.post(f"{api_url}/predict/point", json={
                "latitude": body.latitude, "longitude": body.longitude,
                "location_name": body.name[:120]}, timeout=15)
            if response.status_code in (404, 422):
                raise HTTPException(422, "This coordinate has no valid V3 land coverage. Choose a land location within the prepared Nelson region.")
            response.raise_for_status()
            result = response.json()
            prediction = result["prediction"]
            coordinates = result["location"]["wgs84"]
            score, threshold = prediction["susceptibility_score"], prediction["operational_threshold"]
            if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                    or type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1
                    or prediction.get("score_is_calibrated_probability") is not False
                    or type(prediction.get("susceptible")) is not bool
                    or result["coverage"].get("valid_model_coverage") is not True
                    or result["provenance"].get("model") != "V3 logistic regression"
                    or result["human_review"].get("required") is not True
                    or not math.isclose(float(coordinates["latitude"]), body.latitude, abs_tol=1e-7)
                    or not math.isclose(float(coordinates["longitude"]), body.longitude, abs_tol=1e-7)):
                raise ValueError("Incomplete V3 result.")
            return {"mode": "live_places", "point": result}
        except HTTPException:
            raise
        except requests.RequestException as error:
            raise HTTPException(503, "The V3 prediction API could not be reached. Start Script 30 on this laptop and try again.") from error
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(502, "The prediction API returned an unexpected V3 response. No result was accepted.") from error

    @app.post("/api/places/analyse")
    def area(body: AreaRequest):
        if service.missing_files():
            raise HTTPException(503, "V3 area-analysis raster files are missing. Use your existing complete runtime package in the web project.")
        if not area_lock.acquire(blocking=False):
            raise HTTPException(429, "An area analysis is running. Please wait and try again.", headers={"Retry-After": "3"})
        try:
            # Reconstruct a manual-coordinate record: names sent by the phone do
            # not pretend that the server verified a geocoder match.
            place = service.resolve_place(body.name, body.latitude, body.longitude)
            result = service.analyse(place, body.radius_metres)
            if result.get("status") == "data_unavailable":
                raise HTTPException(422, result.get("reason", "No covered land in the selected area."))
            if result.get("status") != "analysis_ready" or result.get("score_is_calibrated_probability") is not False:
                raise ValueError("Unexpected area result.")
            # Reject nonfinite data before sending JSON to the mobile client.
            json.dumps(result, allow_nan=False)
            map_url = None
            relative = result.get("artifacts", {}).get("local_map")
            if relative:
                map_path = (root / relative).resolve()
                allowed = (root / "outputs" / "place_analysis_v3").resolve()
                if map_path.is_relative_to(allowed) and map_path.is_file() and map_path.suffix == ".png":
                    run_id = str(result["run_id"])
                    maps[run_id] = map_path
                    while len(maps) > 64:
                        maps.popitem(last=False)
                    map_url = f"/api/places/maps/{run_id}"
            # Don't expose laptop filesystem paths or raw workflow state.
            public = {k: v for k, v in result.items() if k != "artifacts"}
            return {"mode": "live_places", "area": public, "map_url": map_url}
        except HTTPException:
            raise
        except Exception as error:
            LOG.exception("Area analysis failed")
            raise HTTPException(503, "Area analysis could not complete. Check the laptop server terminal and try again.") from error
        finally:
            area_lock.release()

    @app.get("/api/places/maps/{run_id}")
    def local_map(run_id: str):
        path = maps.get(run_id)
        if path is None or not path.is_file():
            raise HTTPException(404, "This map is no longer available. Run the area analysis again.")
        return FileResponse(path, media_type="image/png")

    return app


def main():
    root = Path(__file__).resolve().parents[1]
    if not (root / "src" / "climateadapt" / "place_analysis.py").is_file():
        raise SystemExit("Copy this file into your existing ClimateAdapt-AI/scripts folder first.")
    # Headless map rendering, before importing the original service.
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(root / "src"))
    from climateadapt.place_analysis import PlaceAnalysisService, BUILTIN_PLACES, normalise_place_query
    import uvicorn
    app = create_app(PlaceAnalysisService(root), BUILTIN_PLACES, normalise_place_query, root,
                     api_url=os.getenv("CLIMATEADAPT_PREDICTION_URL", "http://127.0.0.1:8000").rstrip("/"),
                     online=os.getenv("CLIMATEADAPT_ONLINE_SEARCH") == "1")
    # One worker only: the search limiter/cache and map registry are process-local.
    uvicorn.run(app, host=os.getenv("CLIMATEADAPT_MOBILE_HOST", "0.0.0.0"),
                port=int(os.getenv("CLIMATEADAPT_MOBILE_PORT", "8081")), workers=1)


if __name__ == "__main__":
    main()
