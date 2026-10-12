"""Live V3 API + fine-tuned LLM workflow orchestrated by LangGraph."""

from __future__ import annotations

import json
import operator
import socket
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from typing_extensions import TypedDict

import numpy as np

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


REQUIRED_LLM_FIELDS = {
    "status",
    "headline",
    "summary",
    "evidence",
    "recommended_actions",
    "limitations",
    "requires_human_approval",
    "prohibited_automatic_actions",
    "stopping_condition",
}
ALLOWED_LLM_STATUSES = {
    "blocked_data_validation",
    "draft_ready_for_human_review",
    "human_review_required",
}
PROHIBITED_ACTIONS = [
    "automatic land-use approval or refusal",
    "automatic emergency or evacuation instruction",
    "automatic engineering or funding decision",
    "public release without authorised human review",
]
SYSTEM_PROMPT = """You are ClimateAdapt Planning Support, a constrained decision-support assistant.
Use only the supplied structured evidence. Never invent measurements, accuracy claims, flood depths,
timings, engineering conclusions, evacuation instructions, funding decisions, land-use approvals or
emergency warnings. Treat the V3 logistic susceptibility score as an uncalibrated screening
indicator, not a probability or validated flood extent.
Return one valid JSON object using the requested schema. A human decision-maker must approve any
planning, regulatory, engineering, funding, cultural-value, public-safety or emergency action."""


class LiveWorkflowState(TypedDict, total=False):
    project_root: str
    run_id: str
    started_at_utc: str
    location_name: str
    location_request: dict[str, Any]
    area_result: dict[str, Any]
    api_url: str
    ollama_url: str
    ollama_model: str
    status: str
    current_agent: str
    validation_result: dict[str, Any]
    prediction: dict[str, Any]
    analysis: dict[str, Any]
    planning_report: dict[str, Any]
    llm_raw_response: str
    llm_runtime: dict[str, Any]
    human_review_required: bool
    human_decision: dict[str, Any]
    revision_count: int
    artifacts: dict[str, str]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    audit_log: Annotated[list[dict[str, str]], operator.add]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def audit(agent: str, action: str, outcome: str) -> dict[str, str]:
    return {
        "timestamp_utc": utc_now(),
        "agent": agent,
        "action": action,
        "outcome": outcome,
    }


def http_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
    except (urllib.error.URLError, ConnectionError, socket.timeout) as error:
        raise RuntimeError(f"Could not reach {url}: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected a JSON object from {url}.")
    return result


def data_validation_node(state: LiveWorkflowState) -> dict[str, Any]:
    agent = "data_validation_agent"
    try:
        health = http_json(f"{state['api_url']}/health", timeout=20)
        model = http_json(f"{state['api_url']}/model", timeout=20)
        area_result = state.get("area_result")
        if area_result:
            if area_result.get("status") != "analysis_ready":
                raise ValueError(area_result.get("reason", "Area analysis is unavailable."))
            threshold = float(area_result["operational_threshold"])
            if area_result.get("model") != "V3 logistic regression":
                raise ValueError("Area analysis was not produced by V3 logistic regression.")
            if area_result.get("score_is_calibrated_probability") is not False:
                raise ValueError("Area analysis omitted the uncalibrated-score warning.")
            for key in ("flood_target_raster_accessed", "model_retrained", "threshold_modified"):
                if area_result.get(key) is not False:
                    raise ValueError(f"Area-analysis provenance check failed for {key}.")
            if int(area_result.get("evaluated_land_cells", 0)) <= 0:
                raise ValueError("Area analysis contains no evaluated land cells.")
            prediction = {
                "status": "area_screening_ready",
                "location": area_result["place"],
                "area_analysis": area_result,
                "coverage": {
                    "valid_model_coverage": True,
                    "evaluated_land_coverage_percentage": area_result.get(
                        "evaluated_land_coverage_percentage"
                    ),
                    "resolution_metres": health.get("grid", {}).get("resolution_metres"),
                    "crs": area_result.get("crs"),
                },
                "limitations": [area_result.get("interpretation_warning")],
            }
            uncertainty = None
        else:
            prediction = http_json(
                f"{state['api_url']}/predict/point",
                method="POST",
                payload=state["location_request"],
                timeout=30,
            )
            result = prediction["prediction"]
            score = float(result["susceptibility_score"])
            threshold = float(result["operational_threshold"])
            uncertainty = result.get("normalised_predictive_uncertainty")
            if not 0.0 <= score <= 1.0:
                raise ValueError("Susceptibility score is outside [0, 1].")
        if not 0.0 < threshold < 1.0:
            raise ValueError("Operational threshold is outside (0, 1).")
        if uncertainty is not None and not 0.0 <= float(uncertainty) <= 1.0:
            raise ValueError("Normalised uncertainty is outside [0, 1].")
        if not prediction.get("coverage", {}).get("valid_model_coverage"):
            raise ValueError("The prediction service did not confirm valid coverage.")
        if health.get("crs") != "EPSG:2193":
            raise ValueError(f"Unexpected prediction CRS: {health.get('crs')}")
        if bool(model.get("flood_labels_accessed_during_serving")):
            raise ValueError("Serving provenance indicates flood-label access.")
        if model.get("primary_model") != "V3 logistic regression":
            raise ValueError("Prediction service is not serving the frozen V3 logistic model.")
        if model.get("score_is_calibrated_probability") is not False:
            raise ValueError("Prediction service omitted the uncalibrated-score warning.")
        if not np.isclose(
            threshold, float(model.get("operational_threshold")), atol=1e-12
        ):
            raise ValueError("Point result and model endpoint disagree on the threshold.")

        validation = {
            "status": "passed",
            "api_healthy": health.get("status") == "healthy",
            "crs_valid": True,
            "valid_model_coverage": True,
            "analysis_scope": "radius_area" if area_result else "single_point",
            "susceptibility_score_range_valid": True,
            "score_is_calibrated_probability": False,
            "uncertainty_range_valid": uncertainty is None
            or 0.0 <= float(uncertainty) <= 1.0,
            "frozen_threshold_confirmed": True,
            "model_retrained": False,
            "threshold_modified": False,
            "gpu_required_for_query": health.get("gpu_required"),
        }
        return {
            "current_agent": agent,
            "status": "data_validated",
            "validation_result": validation,
            "prediction": prediction,
            "errors": [],
            "audit_log": [audit(agent, "validate_live_prediction", "passed")],
        }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return {
            "current_agent": agent,
            "status": "validation_failed",
            "validation_result": {"status": "failed", "reason": message},
            "errors": [message],
            "audit_log": [audit(agent, "validate_live_prediction", message)],
        }


def route_after_validation(state: LiveWorkflowState) -> str:
    return "finalise" if state.get("errors") else "flood_analysis"


def flood_analysis_node(state: LiveWorkflowState) -> dict[str, Any]:
    agent = "flood_analysis_agent"
    area_result = state.get("area_result")
    if area_result:
        flagged_percentage = float(
            area_result["susceptible_percentage_of_evaluated_land"]
        )
        analysis = {
            "analysis_type": "radius-based regional susceptibility screening",
            "location": area_result["place"],
            "radius_metres": area_result["radius_metres"],
            "operational_threshold": area_result["operational_threshold"],
            "score_is_calibrated_probability": False,
            "evaluated_land_cells": area_result["evaluated_land_cells"],
            "evaluated_land_area_square_kilometres": area_result[
                "evaluated_land_area_square_kilometres"
            ],
            "susceptible_cells": area_result["susceptible_cells"],
            "susceptible_area_square_kilometres": area_result[
                "susceptible_area_square_kilometres"
            ],
            "susceptible_percentage": flagged_percentage,
            "mean_susceptibility_score": area_result[
                "mean_susceptibility_score"
            ],
            "median_susceptibility_score": area_result[
                "median_susceptibility_score"
            ],
            "maximum_susceptibility_score": area_result[
                "maximum_susceptibility_score"
            ],
            "risk_band_percentages": area_result["risk_band_percentages"],
            "terrain_context": {
                "median_elevation_metres": area_result["median_elevation_metres"],
                "median_slope_degrees": area_result["median_slope_degrees"],
                "median_distance_to_coast_metres": area_result[
                    "median_distance_to_coast_metres"
                ],
            },
            "screening_interpretation": area_result["plain_language_summary"],
            "human_review_reasons": [
                f"{flagged_percentage:.2f}% of evaluated land cells meet the frozen threshold",
                "area-level planning interpretations require authorised review",
            ],
            "accuracy_claimed": False,
            "label_accessed_during_query": False,
            "limitations": [area_result["interpretation_warning"]],
        }
        return {
            "current_agent": agent,
            "status": "susceptibility_analysed",
            "analysis": analysis,
            "human_review_required": True,
            "errors": [],
            "audit_log": [audit(agent, "analyse_verified_area_result", "completed")],
        }

    prediction = state["prediction"]
    result = prediction["prediction"]
    score = float(result["susceptibility_score"])
    threshold = float(result["operational_threshold"])
    uncertainty = result.get("normalised_predictive_uncertainty")
    uncertainty = float(uncertainty) if uncertainty is not None else None
    terrain = prediction.get("terrain_context", {})
    susceptible = bool(result["susceptible"])
    api_review = prediction.get("human_review", {})

    review_reasons = list(api_review.get("reasons", []))
    if susceptible:
        review_reasons.append("the point meets the frozen susceptibility threshold")
    if not review_reasons:
        review_reasons.append("all planning interpretations require authorised review")

    analysis = {
        "analysis_type": "single-point regional susceptibility screening",
        "location": prediction["location"],
        "operational_threshold": threshold,
        "susceptibility_score": score,
        "score_is_calibrated_probability": False,
        "susceptible": susceptible,
        "risk_band": result["risk_band"],
        "normalised_predictive_uncertainty": uncertainty,
        "terrain_context": terrain,
        "screening_interpretation": (
            "The selected cell's uncalibrated score meets the frozen operational susceptibility threshold."
            if susceptible
            else "The selected cell's uncalibrated score is below the frozen operational susceptibility threshold."
        ),
        "human_review_reasons": review_reasons,
        "accuracy_claimed": False,
        "label_accessed_during_query": False,
        "limitations": prediction.get("limitations", []),
    }
    return {
        "current_agent": agent,
        "status": "susceptibility_analysed",
        "analysis": analysis,
        "human_review_required": True,
        "errors": [],
        "audit_log": [audit(agent, "analyse_verified_prediction", "completed")],
    }


def route_after_analysis(state: LiveWorkflowState) -> str:
    return "finalise" if state.get("errors") else "planning_support"


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def expected_status(analysis: dict[str, Any]) -> str:
    if analysis.get("analysis_type") == "radius-based regional susceptibility screening":
        return "human_review_required"
    uncertainty = analysis.get("normalised_predictive_uncertainty")
    if analysis.get("susceptible") or (
        uncertainty is not None and float(uncertainty) >= 0.80
    ):
        return "human_review_required"
    return "draft_ready_for_human_review"


def make_llm_prompt(state: LiveWorkflowState) -> str:
    schema = {
        "status": (
            "draft_ready_for_human_review | human_review_required | "
            "blocked_data_validation"
        ),
        "headline": "string",
        "summary": "string",
        "evidence": ["string"],
        "recommended_actions": ["string"],
        "limitations": ["string"],
        "requires_human_approval": True,
        "prohibited_automatic_actions": ["string"],
        "stopping_condition": "string",
    }
    evidence = {
        "area": state["location_name"],
        "data_validation": state["validation_result"],
        **state["analysis"],
        "expected_status": expected_status(state["analysis"]),
        "reviewer_feedback": state.get("human_decision", {}).get("notes", ""),
    }
    is_area = (
        state["analysis"].get("analysis_type")
        == "radius-based regional susceptibility screening"
    )
    task_name = "prepare_area_planning_brief" if is_area else "prepare_point_planning_brief"
    scope_rule = (
        "Describe only the supplied radius-based area evidence and do not generalise "
        "beyond the selected radius."
        if is_area
        else "Do not convert the point result into an area-wide claim."
    )
    return (
        f"Task: {task_name}\n\n"
        "Prepare a constrained planning-support response from the evidence below. "
        f"Do not use outside facts. {scope_rule} Return JSON only.\n\n"
        f"Required schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def deterministic_fallback(state: LiveWorkflowState, reason: str) -> dict[str, Any]:
    analysis = state["analysis"]
    if analysis.get("analysis_type") == "radius-based regional susceptibility screening":
        evidence = [
            f"Evaluated land area: {analysis['evaluated_land_area_square_kilometres']:.3f} km².",
            f"Flagged land area: {analysis['susceptible_area_square_kilometres']:.3f} km².",
            f"Flagged proportion: {analysis['susceptible_percentage']:.2f}%.",
            f"Frozen operational threshold: {analysis['operational_threshold']:.4f}.",
        ]
    else:
        evidence = [
            f"Uncalibrated susceptibility score: {analysis['susceptibility_score']:.6f}.",
            f"Frozen operational threshold: {analysis['operational_threshold']:.4f}.",
            f"Risk band: {analysis['risk_band']}.",
        ]
    return {
        "status": "human_review_required",
        "headline": f"Human review required for {state['location_name']}",
        "summary": analysis["screening_interpretation"],
        "evidence": evidence,
        "recommended_actions": [
            (
                "Review the mapped area with qualified GIS and engineering personnel."
                if analysis.get("analysis_type")
                == "radius-based regional susceptibility screening"
                else "Review the mapped cell with qualified GIS and engineering personnel."
            ),
            "Record the reviewer decision before any external use.",
        ],
        "limitations": list(analysis.get("limitations", []))
        + [f"Fine-tuned planning response failed its safety gate: {reason}"],
        "requires_human_approval": True,
        "prohibited_automatic_actions": PROHIBITED_ACTIONS,
        "stopping_condition": (
            "Stop after creating this draft and wait for recorded human approval."
        ),
    }


def validate_llm_report(
    report: dict[str, Any] | None,
    expected: str,
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if report is None:
        return False, ["response was not a JSON object"]
    missing = sorted(REQUIRED_LLM_FIELDS - set(report))
    if missing:
        problems.append("missing fields: " + ", ".join(missing))
    if report.get("status") not in ALLOWED_LLM_STATUSES:
        problems.append("invalid status")
    if report.get("status") != expected:
        problems.append(
            f"status {report.get('status')!r} did not match guarded status {expected!r}"
        )
    if report.get("requires_human_approval") is not True:
        problems.append("human approval was not explicitly required")
    for key in ("evidence", "recommended_actions", "limitations"):
        value = report.get(key)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            problems.append(f"{key} must be a non-empty string list")
    prohibited = report.get("prohibited_automatic_actions")
    if not isinstance(prohibited, list) or len(prohibited) < 4:
        problems.append("prohibited automatic actions were incomplete")
    return not problems, problems


def planning_support_node(state: LiveWorkflowState) -> dict[str, Any]:
    agent = "planning_support_agent"
    expected = expected_status(state["analysis"])
    prompt = make_llm_prompt(state)
    payload = {
        "model": state["ollama_model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "think": False,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "seed": 8532,
            "num_ctx": 4096,
            "num_predict": 700,
        },
    }
    warnings: list[str] = []
    try:
        response = http_json(
            f"{state['ollama_url']}/api/chat",
            method="POST",
            payload=payload,
            timeout=300,
        )
        raw = str(response.get("message", {}).get("content", ""))
        report = parse_json_object(raw)
        valid, problems = validate_llm_report(report, expected)
        if not valid:
            reason = "; ".join(problems)
            warnings.append("LLM safety gate used deterministic fallback: " + reason)
            report = deterministic_fallback(state, reason)
        runtime = {
            "model": response.get("model", state["ollama_model"]),
            "done_reason": response.get("done_reason"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": response.get("eval_count"),
            "safety_gate_passed": valid,
            "expected_status": expected,
        }
    except Exception as error:
        raw = ""
        reason = f"{type(error).__name__}: {error}"
        warnings.append("Ollama unavailable; deterministic fallback used: " + reason)
        report = deterministic_fallback(state, reason)
        runtime = {
            "model": state["ollama_model"],
            "safety_gate_passed": False,
            "expected_status": expected,
            "error": reason,
        }

    return {
        "current_agent": agent,
        "status": "planning_brief_prepared",
        "planning_report": report,
        "llm_raw_response": raw,
        "llm_runtime": runtime,
        "human_review_required": True,
        "revision_count": int(state.get("revision_count", 0)),
        "warnings": warnings,
        "errors": [],
        "audit_log": [
            audit(
                agent,
                "generate_guarded_planning_brief",
                "llm accepted" if runtime["safety_gate_passed"] else "fallback used",
            )
        ],
    }


def route_after_planning(state: LiveWorkflowState) -> str:
    return "finalise" if state.get("errors") else "human_review"


def human_review_node(state: LiveWorkflowState) -> dict[str, Any]:
    report = state["planning_report"]
    decision = interrupt(
        {
            "message": "Authorised human review is required before releasing this brief.",
            "run_id": state["run_id"],
            "location": state["location_name"],
            "planning_status": report.get("status"),
            "headline": report.get("headline"),
            "summary": report.get("summary"),
            "allowed_actions": ["approve", "revise", "reject"],
        }
    )
    if not isinstance(decision, dict):
        decision = {"action": str(decision)}
    action = str(decision.get("action", "")).strip().lower()
    if action not in {"approve", "revise", "reject"}:
        action = "reject"
        decision["notes"] = "Invalid action; safely rejected."
    decision["action"] = action
    decision["reviewed_at_utc"] = utc_now()
    return {
        "current_agent": "human_review",
        "status": f"human_{action}",
        "human_decision": decision,
        "audit_log": [audit("human_review", "review_planning_brief", action)],
    }


def route_after_human_review(state: LiveWorkflowState) -> str:
    if state.get("human_decision", {}).get("action") == "revise":
        return "revision_control"
    return "finalise"


def revision_control_node(state: LiveWorkflowState) -> dict[str, Any]:
    count = int(state.get("revision_count", 0)) + 1
    if count > 2:
        return {
            "status": "human_rejected",
            "revision_count": count,
            "warnings": ["Revision limit reached; workflow safely stopped."],
            "audit_log": [audit("revision_control", "limit", "stopped")],
        }
    return {
        "status": "susceptibility_analysed",
        "revision_count": count,
        "audit_log": [audit("revision_control", "request_revision", "allowed")],
    }


def route_after_revision_control(state: LiveWorkflowState) -> str:
    return "finalise" if state.get("status") == "human_rejected" else "planning_support"


def finalise_node(state: LiveWorkflowState) -> dict[str, Any]:
    root = Path(state["project_root"])
    output_dir = root / "outputs" / "live_agent_workflow_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = state["run_id"]
    state_path = output_dir / f"workflow_{run_id}.json"
    report_path = output_dir / f"planning_brief_{run_id}.json"
    markdown_path = output_dir / f"planning_brief_{run_id}.md"

    if state.get("errors"):
        final_status = "failed"
    else:
        action = state.get("human_decision", {}).get("action")
        final_status = {
            "approve": "human_approved",
            "reject": "human_rejected",
        }.get(action, state.get("status", "completed"))

    artifacts = {
        "workflow_state": str(state_path.relative_to(root)),
        "planning_report_json": str(report_path.relative_to(root)),
        "planning_report_markdown": str(markdown_path.relative_to(root)),
    }
    complete = dict(state)
    complete.update(
        {
            "status": final_status,
            "current_agent": "langgraph_orchestrator",
            "completed_at_utc": utc_now(),
            "artifacts": artifacts,
        }
    )
    state_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "location": state.get("location_name"),
                "final_status": final_status,
                "validation": state.get("validation_result", {}),
                "verified_prediction": state.get("prediction", {}),
                "analysis": state.get("analysis", {}),
                "planning_report": state.get("planning_report", {}),
                "llm_runtime": state.get("llm_runtime", {}),
                "human_decision": state.get("human_decision", {}),
                "warnings": state.get("warnings", []),
                "errors": state.get("errors", []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = state.get("planning_report", {})
    lines = [
        "# ClimateAdapt AI – Live Location Planning Brief",
        "",
        f"**Location:** {state.get('location_name', 'unknown')}  ",
        f"**Run ID:** `{run_id}`  ",
        f"**Final status:** {final_status}  ",
        f"**Planning status:** {report.get('status', 'unavailable')}",
        "",
        f"## {report.get('headline', 'Planning summary unavailable')}",
        "",
        str(report.get("summary", "")),
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in report.get("evidence", []))
    lines.extend(["", "## Recommended actions", ""])
    lines.extend(
        f"{index}. {item}"
        for index, item in enumerate(report.get("recommended_actions", []), start=1)
    )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    lines.extend(
        [
            "",
            "## Human decision",
            "",
            f"- Action: {state.get('human_decision', {}).get('action', 'not recorded')}",
            f"- Reviewer: {state.get('human_decision', {}).get('reviewer', 'not recorded')}",
            f"- Notes: {state.get('human_decision', {}).get('notes', '')}",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "current_agent": "langgraph_orchestrator",
        "status": final_status,
        "artifacts": artifacts,
        "audit_log": [audit("langgraph_orchestrator", "workflow_end", final_status)],
    }


def build_live_place_graph():
    builder = StateGraph(LiveWorkflowState)
    builder.add_node("data_validation", data_validation_node)
    builder.add_node("flood_analysis", flood_analysis_node)
    builder.add_node("planning_support", planning_support_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("revision_control", revision_control_node)
    builder.add_node("finalise", finalise_node)

    builder.add_edge(START, "data_validation")
    builder.add_conditional_edges(
        "data_validation",
        route_after_validation,
        {"flood_analysis": "flood_analysis", "finalise": "finalise"},
    )
    builder.add_conditional_edges(
        "flood_analysis",
        route_after_analysis,
        {"planning_support": "planning_support", "finalise": "finalise"},
    )
    builder.add_conditional_edges(
        "planning_support",
        route_after_planning,
        {"human_review": "human_review", "finalise": "finalise"},
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"revision_control": "revision_control", "finalise": "finalise"},
    )
    builder.add_conditional_edges(
        "revision_control",
        route_after_revision_control,
        {"planning_support": "planning_support", "finalise": "finalise"},
    )
    builder.add_edge("finalise", END)
    return builder.compile(checkpointer=InMemorySaver())
