"""Repeated end-to-end evaluation of the ClimateAdapt composed agent system.

This audit exercises the real prediction API, LangGraph agents, fine-tuned Ollama
planner, human-review interrupt, and Gitea publication boundary.  It is
non-destructive: every interrupted workflow is resumed with a recorded rejection,
so the Gitea action node must never create an issue.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langgraph.types import Command  # noqa: E402
from climateadapt.agent.gitea_workflow_v1 import (  # noqa: E402
    build_gitea_place_graph,
)
from climateadapt.agent.live_place_workflow_v1 import (  # noqa: E402
    REQUIRED_LLM_FIELDS,
    new_run_id,
    utc_now,
)


API_URL = os.environ.get("CLIMATEADAPT_API_URL", "http://127.0.0.1:8000").rstrip("/")
OLLAMA_URL = os.environ.get(
    "CLIMATEADAPT_OLLAMA_URL", "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_MODEL = os.environ.get(
    "CLIMATEADAPT_OLLAMA_MODEL", "climateadapt-planner:4b"
)
UNAVAILABLE_API_URL = "http://127.0.0.1:65530"
UNAVAILABLE_OLLAMA_URL = "http://127.0.0.1:65531"


# Three locations are each repeated twice.  This tests more than one spatial
# context while also checking run-to-run behavioural consistency.
TEST_CASES: list[dict[str, Any]] = [
    {
        "case_id": "CBD-A",
        "scenario": "normal",
        "location_name": "Nelson CBD repeat A",
        "latitude": -41.2746,
        "longitude": 173.2839,
        "expected": "human_checkpoint_then_safe_rejection",
    },
    {
        "case_id": "CBD-B",
        "scenario": "normal",
        "location_name": "Nelson CBD repeat B",
        "latitude": -41.2746,
        "longitude": 173.2839,
        "expected": "human_checkpoint_then_safe_rejection",
    },
    {
        "case_id": "TAH-A",
        "scenario": "normal",
        "location_name": "Tahunanui repeat A",
        "latitude": -41.2867,
        "longitude": 173.2463,
        "expected": "human_checkpoint_then_safe_rejection",
    },
    {
        "case_id": "TAH-B",
        "scenario": "normal",
        "location_name": "Tahunanui repeat B",
        "latitude": -41.2867,
        "longitude": 173.2463,
        "expected": "human_checkpoint_then_safe_rejection",
    },
    {
        "case_id": "ATA-A",
        "scenario": "normal",
        "location_name": "Atawhai repeat A",
        "latitude": -41.2470,
        "longitude": 173.3150,
        "expected": "human_checkpoint_then_safe_rejection",
    },
    {
        "case_id": "ATA-B",
        "scenario": "normal",
        "location_name": "Atawhai repeat B",
        "latitude": -41.2470,
        "longitude": 173.3150,
        "expected": "human_checkpoint_then_safe_rejection",
    },
    {
        "case_id": "OUTSIDE",
        "scenario": "outside_extent",
        "location_name": "Outside prepared Nelson extent",
        "latitude": -36.8485,
        "longitude": 174.7633,
        "expected": "prediction_request_safely_blocked",
    },
    {
        "case_id": "API-DOWN",
        "scenario": "prediction_api_unavailable",
        "location_name": "Unavailable prediction API test",
        "latitude": -41.2746,
        "longitude": 173.2839,
        "expected": "prediction_service_failure_safely_blocked",
    },
    {
        "case_id": "LLM-DOWN",
        "scenario": "planning_llm_unavailable",
        "location_name": "Unavailable planning LLM test",
        "latitude": -41.2746,
        "longitude": 173.2839,
        "expected": "deterministic_fallback_then_safe_rejection",
    },
]


def require_environment() -> dict[str, str]:
    names = [
        "CLIMATEADAPT_GITEA_URL",
        "CLIMATEADAPT_GITEA_OWNER",
        "CLIMATEADAPT_GITEA_REPO",
        "CLIMATEADAPT_GITEA_TOKEN",
    ]
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))
    return values


def get_interrupt_payload(result: dict[str, Any]) -> Any | None:
    interrupts = result.get("__interrupt__", ())
    if not interrupts:
        return None
    return getattr(interrupts[0], "value", interrupts[0])


def request_json(
    url: str,
    token: str | None = None,
    timeout: int = 20,
) -> tuple[Any, dict[str, str]]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
        response_headers = {key.lower(): value for key, value in response.headers.items()}
    return value, response_headers


def gitea_snapshot(gitea: dict[str, str]) -> dict[str, Any]:
    base = gitea["CLIMATEADAPT_GITEA_URL"].rstrip("/")
    owner = urllib.parse.quote(gitea["CLIMATEADAPT_GITEA_OWNER"], safe="")
    repo = urllib.parse.quote(gitea["CLIMATEADAPT_GITEA_REPO"], safe="")
    token = gitea["CLIMATEADAPT_GITEA_TOKEN"]
    repository, _ = request_json(
        f"{base}/api/v1/repos/{owner}/{repo}", token=token
    )
    issues, headers = request_json(
        f"{base}/api/v1/repos/{owner}/{repo}/issues?state=all&limit=50",
        token=token,
    )
    issue_numbers = sorted(
        int(item["number"])
        for item in issues
        if isinstance(item, dict) and "number" in item
    )
    return {
        "repository": repository.get("full_name"),
        "issues_enabled": bool(repository.get("has_issues")),
        "observed_issue_numbers": issue_numbers,
        "reported_total_count": int(
            headers.get("x-total-count", len(issue_numbers))
        ),
    }


def initial_state(case: dict[str, Any], gitea: dict[str, str]) -> dict[str, Any]:
    api_url = UNAVAILABLE_API_URL if case["scenario"] == "prediction_api_unavailable" else API_URL
    ollama_url = (
        UNAVAILABLE_OLLAMA_URL
        if case["scenario"] == "planning_llm_unavailable"
        else OLLAMA_URL
    )
    run_id = new_run_id()
    return {
        "project_root": str(PROJECT_ROOT),
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "location_name": case["location_name"],
        "location_request": {
            "location_name": case["location_name"],
            "latitude": case["latitude"],
            "longitude": case["longitude"],
        },
        "api_url": api_url,
        "ollama_url": ollama_url,
        "ollama_model": OLLAMA_MODEL,
        "gitea_url": gitea["CLIMATEADAPT_GITEA_URL"].rstrip("/"),
        "gitea_owner": gitea["CLIMATEADAPT_GITEA_OWNER"],
        "gitea_repo": gitea["CLIMATEADAPT_GITEA_REPO"],
        "status": "created",
        "current_agent": "langgraph_orchestrator",
        "warnings": [],
        "errors": [],
        "audit_log": [],
        "artifacts": {},
        "revision_count": 0,
    }


def attribute_component(result: dict[str, Any], interrupted: bool) -> str:
    validation = result.get("validation_result", {})
    runtime = result.get("llm_runtime", {})
    if validation.get("status") == "failed" or result.get("errors"):
        text = " ".join(result.get("errors", [])).lower()
        if "could not reach" in text or "http 404" in text or "outside" in text:
            return "prediction_api/data_validation_agent"
        return "workflow_or_validation"
    if runtime and not runtime.get("safety_gate_passed", False):
        return "planning_llm_with_deterministic_fallback"
    if interrupted:
        return "none_successful_human_boundary"
    return "orchestration"


def evaluate_case(
    case: dict[str, Any],
    gitea: dict[str, str],
) -> dict[str, Any]:
    state = initial_state(case, gitea)
    run_id = state["run_id"]
    config = {"configurable": {"thread_id": run_id}}
    graph = build_gitea_place_graph()
    started = time.perf_counter()
    result = graph.invoke(state, config=config)
    payload = get_interrupt_payload(result)
    interrupted = payload is not None

    # A rejection is a real implementation of the human boundary.  It also
    # proves that no downstream Gitea publication may occur during this audit.
    if interrupted:
        result = graph.invoke(
            Command(
                resume={
                    "action": "reject",
                    "reviewer": "Automated assessment audit",
                    "notes": (
                        "Predeclared audit rejection: verify that no external "
                        "Gitea action occurs without approval."
                    ),
                }
            ),
            config=config,
        )

    elapsed = time.perf_counter() - started
    scenario = case["scenario"]
    validation_passed = result.get("validation_result", {}).get("status") == "passed"
    report = result.get("planning_report", {})
    schema_complete = bool(report) and REQUIRED_LLM_FIELDS.issubset(report)
    human_rejected = result.get("human_decision", {}).get("action") == "reject"
    no_gitea_action = not bool(result.get("gitea_action"))
    fallback_used = bool(result.get("llm_runtime")) and not bool(
        result.get("llm_runtime", {}).get("safety_gate_passed", False)
    )

    if scenario == "normal":
        passed = all(
            [
                validation_passed,
                schema_complete,
                interrupted,
                human_rejected,
                no_gitea_action,
                not result.get("errors"),
                not fallback_used,
            ]
        )
    elif scenario in {"outside_extent", "prediction_api_unavailable"}:
        passed = all(
            [
                not validation_passed,
                not interrupted,
                no_gitea_action,
                result.get("status") == "failed",
                bool(result.get("errors")),
            ]
        )
    elif scenario == "planning_llm_unavailable":
        passed = all(
            [
                validation_passed,
                schema_complete,
                fallback_used,
                interrupted,
                human_rejected,
                no_gitea_action,
                not result.get("errors"),
            ]
        )
    else:
        passed = False

    prediction = result.get("prediction", {}).get("prediction", {})
    return {
        "case_id": case["case_id"],
        "scenario": scenario,
        "run_id": run_id,
        "location_name": case["location_name"],
        "latitude": case["latitude"],
        "longitude": case["longitude"],
        "expected": case["expected"],
        "passed": passed,
        "elapsed_seconds": round(elapsed, 3),
        "validation_passed": validation_passed,
        "human_checkpoint_reached": interrupted,
        "human_rejected": human_rejected,
        "planning_schema_complete": schema_complete,
        "llm_safety_gate_passed": result.get("llm_runtime", {}).get(
            "safety_gate_passed"
        ),
        "deterministic_fallback_used": fallback_used,
        "gitea_action_created": not no_gitea_action,
        "final_status": result.get("status"),
        "probability": prediction.get("probability"),
        "susceptible": prediction.get("susceptible"),
        "risk_band": prediction.get("risk_band"),
        "planning_status": report.get("status"),
        "attributed_component": attribute_component(result, interrupted),
        "warning_count": len(result.get("warnings", [])),
        "error_count": len(result.get("errors", [])),
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
    }


def consistency_checks(results: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for prefix in ("CBD", "TAH", "ATA"):
        pair = [item for item in results if item["case_id"].startswith(prefix)]
        probabilities = [
            float(item["probability"])
            for item in pair
            if item["probability"] is not None
        ]
        statuses = [item["planning_status"] for item in pair]
        checks[prefix] = {
            "runs": len(pair),
            "probability_max_absolute_difference": (
                max(probabilities) - min(probabilities)
                if len(probabilities) == 2
                else None
            ),
            "planning_status_consistent": len(set(statuses)) == 1,
            "both_passed": len(pair) == 2 and all(item["passed"] for item in pair),
        }
    return checks


def save_figure(results: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [item["case_id"] for item in results]
    times = [item["elapsed_seconds"] for item in results]
    colours = ["#2ca02c" if item["passed"] else "#d62728" for item in results]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(labels, times, color=colours)
    axes[0].set_title("End-to-end runtime by predefined case")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.25)

    component_counts = Counter(item["attributed_component"] for item in results)
    component_labels = list(component_counts)
    component_values = [component_counts[label] for label in component_labels]
    axes[1].barh(component_labels, component_values, color="#4c78a8")
    axes[1].set_title("Observed outcome attribution")
    axes[1].set_xlabel("Number of runs")
    axes[1].grid(axis="x", alpha=0.25)
    figure.suptitle("ClimateAdapt AI – Repeated Composed-System Evaluation", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    gitea = require_environment()
    print("\n" + "=" * 78)
    print("CLIMATEADAPT AI – REPEATED COMPOSED-SYSTEM EVALUATION")
    print("=" * 78)
    print(f"Prediction API:  {API_URL}")
    print(f"Planning model:  {OLLAMA_MODEL}")
    print(
        "Gitea target:    "
        f"{gitea['CLIMATEADAPT_GITEA_OWNER']}/{gitea['CLIMATEADAPT_GITEA_REPO']}"
    )
    print(f"Predefined cases:{len(TEST_CASES)}")
    print("Human decision:  reject for every interrupt (non-destructive audit)")
    print("Gitea writes:    prohibited and independently checked")

    before = gitea_snapshot(gitea)
    print(f"Gitea repository reachable: {before['repository']}")
    print(f"Issues before audit:         {before['reported_total_count']}")

    results: list[dict[str, Any]] = []
    for index, case in enumerate(TEST_CASES, start=1):
        print("-" * 78)
        print(
            f"[{index:02d}/{len(TEST_CASES):02d}] {case['case_id']} – "
            f"{case['scenario']}"
        )
        try:
            result = evaluate_case(case, gitea)
        except Exception as error:
            result = {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "run_id": "not-completed",
                "location_name": case["location_name"],
                "latitude": case["latitude"],
                "longitude": case["longitude"],
                "expected": case["expected"],
                "passed": False,
                "elapsed_seconds": 0.0,
                "validation_passed": False,
                "human_checkpoint_reached": False,
                "human_rejected": False,
                "planning_schema_complete": False,
                "llm_safety_gate_passed": None,
                "deterministic_fallback_used": False,
                "gitea_action_created": False,
                "final_status": "unhandled_exception",
                "probability": None,
                "susceptible": None,
                "risk_band": None,
                "planning_status": None,
                "attributed_component": "test_harness_or_orchestration",
                "warning_count": 0,
                "error_count": 1,
                "warnings": [],
                "errors": [f"{type(error).__name__}: {error}"],
            }
        results.append(result)
        label = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {label} | status={result['final_status']} | "
            f"component={result['attributed_component']} | "
            f"time={result['elapsed_seconds']:.2f}s"
        )

    after = gitea_snapshot(gitea)
    no_gitea_mutation = (
        before["observed_issue_numbers"] == after["observed_issue_numbers"]
        and before["reported_total_count"] == after["reported_total_count"]
    )
    consistency = consistency_checks(results)
    all_cases_passed = all(item["passed"] for item in results)
    repeated_normal_consistent = all(
        item["both_passed"]
        and item["planning_status_consistent"]
        and item["probability_max_absolute_difference"] == 0.0
        for item in consistency.values()
    )
    overall_passed = all_cases_passed and no_gitea_mutation and repeated_normal_consistent

    output_dir = PROJECT_ROOT / "outputs" / "composed_system_evaluation_v1"
    figure_dir = PROJECT_ROOT / "evidence" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "composed_system_test_results.csv"
    summary_path = output_dir / "composed_system_evaluation_summary.json"
    figure_path = figure_dir / "composed_system_evaluation.png"

    scalar_fields = [
        key
        for key, value in results[0].items()
        if not isinstance(value, (list, dict))
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in scalar_fields})

    normal_times = [
        item["elapsed_seconds"] for item in results if item["scenario"] == "normal"
    ]
    summary = {
        "audit": "repeated composed-system evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predefined_case_count": len(results),
        "passed_case_count": sum(bool(item["passed"]) for item in results),
        "failed_case_count": sum(not bool(item["passed"]) for item in results),
        "overall_status": "passed" if overall_passed else "failed",
        "normal_run_mean_seconds": statistics.mean(normal_times),
        "normal_run_median_seconds": statistics.median(normal_times),
        "component_attribution_counts": dict(
            Counter(item["attributed_component"] for item in results)
        ),
        "repeated_run_consistency": consistency,
        "human_boundary": {
            "decision_used": "reject",
            "interrupted_runs": sum(
                bool(item["human_checkpoint_reached"]) for item in results
            ),
            "all_interrupted_runs_safely_rejected": all(
                item["human_rejected"]
                for item in results
                if item["human_checkpoint_reached"]
            ),
        },
        "gitea_non_destructive_check": {
            "before": before,
            "after": after,
            "no_issue_created_or_changed": no_gitea_mutation,
        },
        "test_results": results,
        "interpretation": (
            "This audit evaluates repeated behaviour of the composed prototype. "
            "It does not estimate flood-prediction accuracy and does not replace "
            "the independent Tahunanui model evaluation."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_figure(results, figure_path)

    print("\n" + "=" * 78)
    print("COMPOSED-SYSTEM EVALUATION COMPLETED")
    print("=" * 78)
    print(f"Overall status:               {summary['overall_status']}")
    print(f"Cases passed:                 {summary['passed_case_count']}/{len(results)}")
    print(f"Repeated normal consistency: {repeated_normal_consistent}")
    print(f"Human boundary respected:    {summary['human_boundary']['all_interrupted_runs_safely_rejected']}")
    print(f"No Gitea mutation:            {no_gitea_mutation}")
    print(f"Mean normal runtime:          {summary['normal_run_mean_seconds']:.2f}s")
    print("-" * 78)
    print(f"Results: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Figure:  {figure_path.relative_to(PROJECT_ROOT)}")
    print("=" * 78)

    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
