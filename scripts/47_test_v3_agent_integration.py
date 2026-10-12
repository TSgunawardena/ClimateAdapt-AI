"""Non-destructive V3 API-to-agent integration test.

This test runs only the data-validation and flood-analysis nodes. It does not
call Ollama, pause for human review, write Gitea issues or save workflow files.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.agent.live_place_workflow_v1 import (  # noqa: E402
    data_validation_node,
    flood_analysis_node,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    state = {
        "project_root": str(PROJECT_ROOT),
        "run_id": "v3-agent-integration-test",
        "location_name": "Nelson V3 integration test",
        "location_request": {
            "latitude": -41.29,
            "longitude": 173.24,
            "location_name": "Nelson V3 integration test",
        },
        "api_url": "http://127.0.0.1:8000",
        "warnings": [],
        "errors": [],
        "audit_log": [],
    }

    print("=" * 76)
    print("CLIMATEADAPT V3 – API TO AGENT INTEGRATION TEST")
    print("=" * 76)
    validation_update = data_validation_node(state)
    require(not validation_update.get("errors"), str(validation_update.get("errors")))
    require(validation_update.get("status") == "data_validated", "Validation failed")
    validation = validation_update["validation_result"]
    require(validation.get("susceptibility_score_range_valid") is True, "Score invalid")
    require(validation.get("score_is_calibrated_probability") is False, "Score warning missing")
    print("PASS  Data Validation Agent accepted the frozen V3 service")

    merged = {**state, **validation_update}
    analysis_update = flood_analysis_node(merged)
    require(not analysis_update.get("errors"), str(analysis_update.get("errors")))
    analysis = analysis_update["analysis"]
    require("susceptibility_score" in analysis, "Analysis omitted V3 score")
    require("probability" not in analysis, "Analysis retained deprecated probability wording")
    require(analysis.get("score_is_calibrated_probability") is False, "Calibration warning missing")
    require(analysis_update.get("human_review_required") is True, "Human boundary missing")
    print("PASS  Flood Analysis Agent uses susceptibility-score semantics")
    print("PASS  Human-review requirement remains active")
    print("-" * 76)
    print(f"Score:     {analysis['susceptibility_score']:.6f}")
    print(f"Threshold: {analysis['operational_threshold']:.4f}")
    print(f"Flagged:   {analysis['susceptible']}")
    print("=" * 76)
    print("RESULT: V3 AGENT INTEGRATION PASSED")
    print("No LLM or Gitea action was performed.")


if __name__ == "__main__":
    main()
