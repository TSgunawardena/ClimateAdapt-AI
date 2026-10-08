"""Non-destructive robustness and safe-failure audit for ClimateAdapt AI.

Prerequisites: the Script 30 API, Ollama and Gitea must be running. The audit
sends invalid and valid prediction requests, checks simulated service outages,
verifies the Gitea approval gate without publishing anything, and scans saved
Gitea workflow evidence for accidental token disclosure.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climateadapt.agent.gitea_workflow_v1 import (  # noqa: E402
    gitea_publish_node,
    gitea_request,
    route_after_human_review_for_gitea,
)


OUTPUT_DIR = ROOT / "outputs" / "robustness_audit_v1"
FIGURE_DIR = ROOT / "evidence" / "figures"
RESULTS_CSV = OUTPUT_DIR / "robustness_test_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "robustness_audit_summary.json"
FIGURE_PATH = FIGURE_DIR / "robustness_audit.png"

API_URL = os.getenv("CLIMATEADAPT_API_URL", "http://127.0.0.1:8000").rstrip("/")
OLLAMA_URL = os.getenv("CLIMATEADAPT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("CLIMATEADAPT_OLLAMA_MODEL", "climateadapt-planner:4b")


def http_json(
    url: str,
    method: str = "GET",
    payload: Any | None = None,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 8.0,
) -> tuple[int | None, Any, str | None]:
    body = raw_body
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text
            return int(response.status), value, None
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        return int(error.code), value, None
    except Exception as error:  # connection failure is an expected test outcome
        return None, None, f"{type(error).__name__}: {error}"


def compact_detail(value: Any, limit: int = 180) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    def record(
        test_id: str,
        category: str,
        description: str,
        passed: bool,
        expected: str,
        observed: str,
        control: str,
    ) -> None:
        results.append(
            {
                "test_id": test_id,
                "category": category,
                "description": description,
                "passed": bool(passed),
                "expected": expected,
                "observed": observed,
                "safety_control": control,
            }
        )
        print(f"[{test_id}] {'PASS' if passed else 'FAIL'} – {description}")
        print(f"       expected={expected} | observed={observed}")

    print("\n" + "=" * 78)
    print("CLIMATEADAPT AI – NON-DESTRUCTIVE ROBUSTNESS AND SAFE-FAILURE AUDIT")
    print("=" * 78)
    print(f"Prediction API: {API_URL}")
    print(f"Ollama model:   {OLLAMA_MODEL}")
    print("Gitea mode:     read-only repository check; no issue will be created")
    print("Secrets:        never written to the report\n")

    status, body, error = http_json(f"{API_URL}/health")
    healthy = status == 200 and isinstance(body, dict) and body.get("status") == "healthy"
    record(
        "API-01", "availability", "Prediction API health and raster readiness",
        healthy, "HTTP 200 and healthy", error or f"HTTP {status}: {compact_detail(body)}",
        "Startup health gate",
    )

    status, body, error = http_json(
        f"{API_URL}/predict/point",
        method="POST",
        payload={"latitude": -41.29, "longitude": 173.24, "location_name": "Robustness audit"},
    )
    valid_ready = (
        status == 200
        and isinstance(body, dict)
        and body.get("prediction", {}).get("operational_threshold") is not None
        and body.get("coverage", {}).get("valid_model_coverage") is True
    )
    record(
        "API-02", "valid_request", "Known Nelson coordinate returns provenance-rich result",
        valid_ready, "HTTP 200 with coverage and threshold", error or f"HTTP {status}: {compact_detail(body)}",
        "Validated regional raster lookup",
    )

    invalid_cases = [
        ("API-03", "No coordinate pair", {}, {422}),
        ("API-04", "Incomplete WGS84 pair", {"latitude": -41.29}, {422}),
        (
            "API-05",
            "Both coordinate systems supplied",
            {"latitude": -41.29, "longitude": 173.24, "easting": 1620000, "northing": 5430000},
            {422},
        ),
        ("API-06", "Out-of-range latitude", {"latitude": 120, "longitude": 173.24}, {422}),
        ("API-07", "Coordinate outside prepared Nelson extent", {"latitude": 0, "longitude": 0}, {404}),
    ]
    for test_id, description, payload, expected_codes in invalid_cases:
        status, body, error = http_json(
            f"{API_URL}/predict/point", method="POST", payload=payload
        )
        record(
            test_id, "input_validation", description,
            status in expected_codes, f"HTTP {sorted(expected_codes)} rejection",
            error or f"HTTP {status}: {compact_detail(body)}",
            "Pydantic and geographic boundary validation",
        )

    status, body, error = http_json(
        f"{API_URL}/predict/point", method="POST", raw_body=b'{"latitude":'
    )
    record(
        "API-08", "input_validation", "Malformed JSON is rejected safely",
        status == 422, "HTTP 422", error or f"HTTP {status}: {compact_detail(body)}",
        "Request parser validation",
    )

    status, body, error = http_json(f"{API_URL}/predict/point", method="GET")
    record(
        "API-09", "interface", "Unsupported HTTP method is rejected",
        status == 405, "HTTP 405", error or f"HTTP {status}: {compact_detail(body)}",
        "Explicit API method boundary",
    )

    status, _, error = http_json("http://127.0.0.1:65534/health", timeout=1.0)
    record(
        "SVC-01", "dependency_failure", "Unavailable prediction service is detected",
        status is None and error is not None, "Caught connection failure",
        error or f"unexpected HTTP {status}", "Bounded timeout and exception handling",
    )

    status, body, error = http_json(f"{OLLAMA_URL}/api/tags")
    model_names = []
    if status == 200 and isinstance(body, dict):
        model_names = [str(item.get("name", "")) for item in body.get("models", [])]
    ollama_ready = status == 200 and OLLAMA_MODEL in model_names
    record(
        "LLM-01", "availability", "Fine-tuned planning model is available in Ollama",
        ollama_ready, f"HTTP 200 and {OLLAMA_MODEL} listed",
        error or f"HTTP {status}; models={model_names}", "Model availability preflight",
    )

    status, _, error = http_json("http://127.0.0.1:65533/api/tags", timeout=1.0)
    record(
        "LLM-02", "dependency_failure", "Unavailable LLM service is detected",
        status is None and error is not None, "Caught connection failure",
        error or f"unexpected HTTP {status}", "Bounded timeout and exception handling",
    )

    route_checks = {
        "approve": route_after_human_review_for_gitea({"human_decision": {"action": "approve"}}),
        "revise": route_after_human_review_for_gitea({"human_decision": {"action": "revise"}}),
        "reject": route_after_human_review_for_gitea({"human_decision": {"action": "reject"}}),
    }
    expected_routes = {
        "approve": "gitea_publish", "revise": "revision_control", "reject": "finalise"
    }
    record(
        "WF-01", "human_control", "Human decisions route to the correct workflow branches",
        route_checks == expected_routes, str(expected_routes), str(route_checks),
        "LangGraph conditional approval gate",
    )

    blocked = gitea_publish_node(
        {
            "human_decision": {"action": "reject"},
            "audit_log": [],
            "errors": [],
        }
    )
    blocked_ok = (
        blocked.get("status") == "gitea_publish_blocked"
        and bool(blocked.get("errors"))
    )
    record(
        "WF-02", "human_control", "Rejected workflow cannot publish a Gitea issue",
        blocked_ok, "gitea_publish_blocked with recorded error",
        compact_detail(blocked), "Publication node rechecks recorded approval",
    )

    gitea_url = os.getenv("CLIMATEADAPT_GITEA_URL", "").rstrip("/")
    owner = os.getenv("CLIMATEADAPT_GITEA_OWNER", "")
    repo = os.getenv("CLIMATEADAPT_GITEA_REPO", "")
    token = os.getenv("CLIMATEADAPT_GITEA_TOKEN", "")
    if gitea_url and owner and repo and token:
        owner_path = urllib.parse.quote(owner, safe="")
        repo_path = urllib.parse.quote(repo, safe="")
        status, body, error = http_json(
            f"{gitea_url}/api/v1/repos/{owner_path}/{repo_path}",
            headers={"Authorization": f"token {token}"},
        )
        expected_name = f"{owner}/{repo}"
        gitea_ok = (
            status == 200
            and isinstance(body, dict)
            and body.get("full_name") == expected_name
            and body.get("has_issues") is True
        )
        observed = error or f"HTTP {status}; repository={body.get('full_name') if isinstance(body, dict) else body}"
    else:
        gitea_ok = False
        observed = "One or more CLIMATEADAPT_GITEA_* environment variables are missing"
    record(
        "GIT-01", "external_tool", "Gitea repository is reachable with Issues enabled",
        gitea_ok, "Read-only HTTP 200 with expected repository", observed,
        "Authenticated read-only preflight",
    )

    saved_token = os.environ.pop("CLIMATEADAPT_GITEA_TOKEN", None)
    try:
        missing_token_blocked = False
        try:
            gitea_request(
                {"gitea_url": gitea_url or "http://127.0.0.1:3000"},
                "/api/v1/version",
            )
        except RuntimeError as error:
            missing_token_blocked = "not configured" in str(error)
    finally:
        if saved_token is not None:
            os.environ["CLIMATEADAPT_GITEA_TOKEN"] = saved_token
    record(
        "GIT-02", "secret_handling", "Missing Gitea token blocks requests before network access",
        missing_token_blocked, "RuntimeError before request", "blocked" if missing_token_blocked else "not blocked",
        "Environment-only credential precondition",
    )

    evidence_dir = ROOT / "outputs" / "gitea_agent_workflow_v1"
    scanned_files = 0
    leaks = 0
    if saved_token and len(saved_token) >= 8 and evidence_dir.exists():
        for path in evidence_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".md", ".csv", ".txt"}:
                scanned_files += 1
                text = path.read_text(encoding="utf-8", errors="ignore")
                if saved_token in text:
                    leaks += 1
    token_scan_ok = bool(saved_token) and scanned_files > 0 and leaks == 0
    record(
        "GIT-03", "secret_handling", "Saved workflow evidence contains no Gitea token",
        token_scan_ok, "At least one file scanned and zero leaks",
        f"files_scanned={scanned_files}; leaks={leaks}", "Token excluded from workflow state and artifacts",
    )

    frame = pd.DataFrame(results)
    frame.to_csv(RESULTS_CSV, index=False)
    passed_count = int(frame["passed"].sum())
    failed_count = int((~frame["passed"]).sum())
    all_passed = failed_count == 0
    elapsed = time.perf_counter() - started
    summary = {
        "audit": "ClimateAdapt non-destructive robustness and safe-failure audit",
        "status": "passed" if all_passed else "attention_required",
        "tests": len(frame),
        "passed": passed_count,
        "failed": failed_count,
        "gitea_write_operations_performed": 0,
        "model_weights_modified": False,
        "threshold_modified": False,
        "secrets_written_to_outputs": False,
        "processing_time_seconds": elapsed,
        "failed_test_ids": frame.loc[~frame["passed"], "test_id"].tolist(),
        "interpretation": (
            "All tested boundaries failed safely and required services were available."
            if all_passed
            else "One or more controls need attention; inspect the CSV before submission."
        ),
        "outputs": {
            "results_csv": str(RESULTS_CSV.relative_to(ROOT)),
            "figure": str(FIGURE_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    category = frame.groupby("category")["passed"].agg(["sum", "count"]).sort_index()
    figure, axis = plt.subplots(figsize=(11, 6))
    positions = range(len(category))
    axis.bar(positions, category["count"], color="#d9d9d9", label="Total")
    axis.bar(positions, category["sum"], color="#2ca25f", label="Passed")
    axis.set_xticks(list(positions), category.index, rotation=30, ha="right")
    axis.set_ylim(0, max(category["count"].max() + 1, 3))
    axis.set_ylabel("Number of tests")
    axis.set_title("ClimateAdapt AI – Robustness and Safe-Failure Audit")
    axis.legend()
    for index, (passed_value, total_value) in enumerate(
        zip(category["sum"].to_numpy(), category["count"].to_numpy())
    ):
        axis.text(
            index,
            total_value + 0.08,
            f"{int(passed_value)}/{int(total_value)}",
            ha="center",
        )
    figure.text(
        0.5, 0.015,
        "Non-destructive audit: no Gitea issues created, no weights or thresholds changed.",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("\n" + "=" * 78)
    print("ROBUSTNESS AUDIT COMPLETED")
    print("=" * 78)
    print(f"Final status:  {summary['status']}")
    print(f"Tests:         {len(frame)}")
    print(f"Passed:        {passed_count}")
    print(f"Failed:        {failed_count}")
    print("Gitea writes:  0")
    print("Secrets saved: 0")
    print("-" * 78)
    print(f"Results: {RESULTS_CSV.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Figure:  {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Time:    {elapsed:.2f} seconds")
    print("=" * 78)
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
