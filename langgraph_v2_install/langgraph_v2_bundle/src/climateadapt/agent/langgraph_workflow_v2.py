"""LangGraph orchestration, conditional routing and human review."""

from __future__ import annotations

import json
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .data_validation_agent_v2 import data_validation_node
from .flood_analysis_agent_v2 import flood_analysis_node
from .planning_support_agent_v2 import planning_support_node
from .state_v2 import ClimateAdaptGraphState, audit_event, utc_now


def route_after_validation(state: ClimateAdaptGraphState) -> str:
    return "finalise" if state.get("errors") else "flood_analysis"


def route_after_analysis(state: ClimateAdaptGraphState) -> str:
    return "finalise" if state.get("errors") else "planning_support"


def route_after_planning(state: ClimateAdaptGraphState) -> str:
    if state.get("errors"):
        return "finalise"
    return "human_review" if state.get("human_review_required") else "finalise"


def human_review_node(state: ClimateAdaptGraphState) -> dict:
    planning = state.get("planning_summary", {})
    analysis = state.get("flood_analysis", {})
    decision = interrupt(
        {
            "message": "Human approval is required before releasing the planning brief.",
            "run_id": state["run_id"],
            "area": state["area_name"],
            "priority_level": planning.get("priority_level"),
            "human_review_cells": analysis.get("physicalist_guardrail", {}).get(
                "human_review_cells"
            ),
            "warnings": state.get("warnings", []),
            "allowed_actions": ["approve", "revise", "reject"],
        }
    )
    if not isinstance(decision, dict):
        decision = {"action": str(decision)}
    action = str(decision.get("action", "")).strip().lower()
    if action not in {"approve", "revise", "reject"}:
        action = "reject"
        decision["notes"] = "Invalid review action; safely rejected."
    decision["action"] = action
    decision.setdefault("reviewed_at_utc", utc_now())
    return {
        "current_agent": "human_review",
        "status": f"human_{action}",
        "human_decision": decision,
        "audit_log": [audit_event("human_review", "review", action)],
    }


def route_after_human_review(state: ClimateAdaptGraphState) -> str:
    if state.get("human_decision", {}).get("action") == "revise":
        return "planning_support"
    return "finalise"


def finalise_node(state: ClimateAdaptGraphState) -> dict:
    root = Path(state["project_root"])
    output_dir = root / "outputs" / "agent_workflow_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    if state.get("errors"):
        final_status = "failed"
    else:
        action = state.get("human_decision", {}).get("action")
        final_status = {
            "approve": "human_approved",
            "reject": "human_rejected",
        }.get(action, "completed_without_required_review")

    artifacts = {
        **state.get("artifacts", {}),
        "workflow_state": "outputs/agent_workflow_v2/workflow_state_v2.json",
        "agent_report_json": "outputs/agent_workflow_v2/tahunanui_agent_report_v2.json",
        "agent_report_markdown": "outputs/agent_workflow_v2/tahunanui_agent_report_v2.md",
        "human_review_record": "outputs/agent_workflow_v2/human_review_record_v2.json",
    }
    report = {
        "run_id": state["run_id"],
        "area": state["area_name"],
        "final_status": final_status,
        "completed_at_utc": utc_now(),
        "validation": state.get("validation_summary", {}),
        "multi_paradigm_analysis": state.get("flood_analysis", {}),
        "planning_support": state.get("planning_summary", {}),
        "human_decision": state.get("human_decision", {}),
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
        "audit_log": state.get("audit_log", []),
        "source_artifacts": state.get("paths", {}),
    }

    report_json = output_dir / "tahunanui_agent_report_v2.json"
    report_markdown = output_dir / "tahunanui_agent_report_v2.md"
    review_json = output_dir / "human_review_record_v2.json"
    state_json = output_dir / "workflow_state_v2.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    review_json.write_text(
        json.dumps(state.get("human_decision", {}), indent=2), encoding="utf-8"
    )

    analysis = state.get("flood_analysis", {})
    guardrail = analysis.get("physicalist_guardrail", {})
    uncertainty = analysis.get("bayesian_style_uncertainty", {})
    lines = [
        "# ClimateAdapt AI – LangGraph V2 Planning Brief",
        "",
        f"**Run ID:** `{state['run_id']}`  ",
        f"**Area:** {state['area_name']}  ",
        f"**Final status:** {final_status}  ",
        f"**Priority:** {analysis.get('planning_priority', 'unavailable')}",
        "",
        "## Multi-paradigm evidence",
        "",
        f"- Frozen operational threshold: {analysis.get('operational_threshold', 'n/a')}",
        f"- Original susceptible cells: {guardrail.get('original_candidate_cells', 'n/a')}",
        f"- Physically supported cells: {guardrail.get('physically_supported_cells', 'n/a')}",
        f"- Human-review cells: {guardrail.get('human_review_cells', 'n/a')}",
        f"- Terrain-rule rejections: {guardrail.get('terrain_rejected_cells', 'n/a')}",
        f"- Coast-disconnected rejections: {guardrail.get('coast_disconnected_cells', 'n/a')}",
        f"- Uncertainty/error enrichment: {uncertainty.get('error_enrichment', 'n/a')}",
        "",
        "## Planning recommendations",
        "",
    ]
    for item in state.get("recommendations", []):
        lines.append(f"{item['priority']}. **{item['action']}** – {item['reason']}")
    lines.extend(["", "## Mandatory limitations", ""])
    for limitation in state.get("planning_summary", {}).get(
        "mandatory_limitations", []
    ):
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "## Human decision",
            "",
            f"- Action: {state.get('human_decision', {}).get('action', 'not required')}",
            f"- Reviewer: {state.get('human_decision', {}).get('reviewer', 'not recorded')}",
            f"- Notes: {state.get('human_decision', {}).get('notes', '')}",
        ]
    )
    report_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    final_event = audit_event("langgraph_orchestrator", "workflow_end", final_status)
    snapshot = dict(state)
    snapshot.update(
        {
            "status": final_status,
            "current_agent": "langgraph_orchestrator",
            "artifacts": artifacts,
            "audit_log": state.get("audit_log", []) + [final_event],
        }
    )
    state_json.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return {
        "current_agent": "langgraph_orchestrator",
        "status": final_status,
        "artifacts": artifacts,
        "audit_log": [final_event],
    }


def build_climateadapt_graph():
    builder = StateGraph(ClimateAdaptGraphState)
    builder.add_node("data_validation", data_validation_node)
    builder.add_node("flood_analysis", flood_analysis_node)
    builder.add_node("planning_support", planning_support_node)
    builder.add_node("human_review", human_review_node)
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
        {"planning_support": "planning_support", "finalise": "finalise"},
    )
    builder.add_edge("finalise", END)
    return builder.compile(checkpointer=InMemorySaver())

