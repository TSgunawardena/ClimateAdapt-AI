"""Smoke-test the V3 radius-based place-analysis service and agent handoff."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.place_analysis import PlaceAnalysisService  # noqa: E402
from climateadapt.agent.live_place_workflow_v1 import (  # noqa: E402
    data_validation_node,
    flood_analysis_node,
    new_run_id,
    utc_now,
)


def main() -> None:
    print("=" * 76)
    print("CLIMATEADAPT V3 – AREA ANALYSIS AND AGENT HANDOFF TEST")
    print("=" * 76)
    service = PlaceAnalysisService(PROJECT_ROOT)
    missing = service.missing_files()
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  {path.relative_to(PROJECT_ROOT)}")
        raise SystemExit(1)

    monaco = service.resolve_place("Monaco, Nelson")
    assert monaco["source"] == "offline_nelson_catalogue"
    assert monaco["display_name"] == "Monaco, Nelson"

    rocks_road = service.resolve_place("Rocks Road")
    print("Analysing Rocks Road with a 1 kilometre radius...")
    area = service.analyse(rocks_road, 1000)
    assert area["status"] == "analysis_ready", area
    assert area["model"] == "V3 logistic regression"
    assert area["score_is_calibrated_probability"] is False
    assert area["flood_target_raster_accessed"] is False
    assert area["model_retrained"] is False
    assert area["threshold_modified"] is False
    assert area["operational_threshold"] == 0.004
    assert area["evaluated_land_cells"] > 0
    assert 0.0 <= area["susceptible_percentage_of_evaluated_land"] <= 100.0
    assert "mean_probability" not in area
    for artifact in area["artifacts"].values():
        assert (PROJECT_ROOT / artifact).exists(), artifact

    state = {
        "project_root": str(PROJECT_ROOT),
        "run_id": new_run_id(),
        "started_at_utc": utc_now(),
        "location_name": rocks_road["display_name"],
        "location_request": {
            "location_name": rocks_road["display_name"],
            "latitude": rocks_road["latitude"],
            "longitude": rocks_road["longitude"],
        },
        "area_result": area,
        "api_url": "http://127.0.0.1:8000",
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "climateadapt-planner:4b",
        "status": "created",
        "current_agent": "test",
        "warnings": [],
        "errors": [],
        "audit_log": [],
        "artifacts": {},
        "revision_count": 0,
    }
    validation_update = data_validation_node(state)
    assert validation_update["status"] == "data_validated", validation_update
    state.update(validation_update)
    analysis_update = flood_analysis_node(state)
    assert analysis_update["status"] == "susceptibility_analysed"
    assert analysis_update["analysis"]["analysis_type"].startswith("radius-based")
    assert analysis_update["human_review_required"] is True

    print("PASS  Monaco resolves to the fixed Nelson catalogue entry")
    print("PASS  Rocks Road produces a V3 radius-based area result")
    print("PASS  Result uses uncalibrated susceptibility-score semantics")
    print("PASS  Data Validation Agent accepts the V3 area evidence")
    print("PASS  Flood Analysis Agent interprets area-level evidence")
    print("-" * 76)
    print(
        json.dumps(
            {
                "evaluated_land_km2": area[
                    "evaluated_land_area_square_kilometres"
                ],
                "flagged_land_km2": area[
                    "susceptible_area_square_kilometres"
                ],
                "flagged_percentage": area[
                    "susceptible_percentage_of_evaluated_land"
                ],
                "threshold": area["operational_threshold"],
                "map": area["artifacts"]["local_map"],
            },
            indent=2,
        )
    )
    print("=" * 76)
    print("RESULT: V3 AREA ANALYSIS PASSED")
    print("No LLM, Gitea action, model training or threshold change was performed.")


if __name__ == "__main__":
    main()
