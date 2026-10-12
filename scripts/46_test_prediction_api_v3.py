"""Non-destructive smoke test for the ClimateAdapt V3 prediction API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


def request_json(path: str, method: str = "GET", payload: dict | None = None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = json.loads(error.read().decode("utf-8"))
        return error.code, body


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    print("=" * 72)
    print("CLIMATEADAPT V3 PREDICTION API SMOKE TEST")
    print("=" * 72)

    status, health = request_json("/health")
    require(status == 200 and health.get("status") == "healthy", "Health check failed")
    require("susceptibility_score" in health.get("rasters_open", []), "V3 score raster missing")
    print("PASS  /health – V3 rasters are open")

    status, model = request_json("/model")
    require(status == 200, "Model endpoint failed")
    require(model.get("primary_model") == "V3 logistic regression", "Wrong primary model")
    require(model.get("score_is_calibrated_probability") is False, "Calibration warning missing")
    require(abs(float(model.get("operational_threshold")) - 0.004) < 1e-9, "Wrong threshold")
    print("PASS  /model – frozen V3 logistic provenance is correct")

    status, result = request_json(
        "/predict/point",
        method="POST",
        payload={
            "latitude": -41.29,
            "longitude": 173.24,
            "location_name": "V3 API smoke test",
        },
    )
    require(status == 200, f"Known Nelson request failed: {result}")
    prediction = result.get("prediction", {})
    require("susceptibility_score" in prediction, "V3 score missing from response")
    require(prediction.get("score_is_calibrated_probability") is False, "Score warning missing")
    require(result.get("provenance", {}).get("model") == "V3 logistic regression", "Wrong provenance")
    require(result.get("coverage", {}).get("valid_model_coverage") is True, "Coverage missing")
    print("PASS  /predict/point – known Nelson coordinate")

    status, _ = request_json("/predict/point", method="POST", payload={})
    require(status == 422, "Empty coordinate request was not rejected")
    print("PASS  malformed request rejected with HTTP 422")

    status, _ = request_json(
        "/predict/point",
        method="POST",
        payload={"latitude": 0.0, "longitude": 0.0},
    )
    require(status == 404, "Outside-region coordinate was not rejected")
    print("PASS  outside-region request rejected with HTTP 404")

    print("=" * 72)
    print("RESULT: ALL V3 API SMOKE TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
