"""LangGraph workflow that publishes approved briefs and verifies Gitea state."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .live_place_workflow_v1 import (
    LiveWorkflowState,
    audit,
    data_validation_node,
    flood_analysis_node,
    human_review_node,
    planning_support_node,
    revision_control_node,
    route_after_analysis,
    route_after_planning,
    route_after_revision_control,
    route_after_validation,
    utc_now,
)


class GiteaWorkflowState(LiveWorkflowState, total=False):
    gitea_url: str
    gitea_owner: str
    gitea_repo: str
    gitea_action: dict[str, Any]
    gitea_verification: dict[str, Any]


def gitea_request(
    state: GiteaWorkflowState,
    endpoint: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    token = os.environ.get("CLIMATEADAPT_GITEA_TOKEN", "").strip()
    if not token:
        raise RuntimeError("CLIMATEADAPT_GITEA_TOKEN is not configured.")

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{state['gitea_url']}{endpoint}",
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gitea returned HTTP {error.code} for {endpoint}: {detail}"
        ) from error
    except (urllib.error.URLError, ConnectionError, socket.timeout) as error:
        raise RuntimeError(f"Could not reach Gitea endpoint {endpoint}: {error}") from error

    if not isinstance(value, dict):
        raise RuntimeError("Gitea did not return the expected JSON object.")
    return value


def repository_path(state: GiteaWorkflowState) -> str:
    owner = urllib.parse.quote(state["gitea_owner"], safe="")
    repo = urllib.parse.quote(state["gitea_repo"], safe="")
    return f"/api/v1/repos/{owner}/{repo}"


def route_after_human_review_for_gitea(state: GiteaWorkflowState) -> str:
    action = state.get("human_decision", {}).get("action")
    if action == "approve":
        return "gitea_publish"
    if action == "revise":
        return "revision_control"
    return "finalise"


def build_issue_body(state: GiteaWorkflowState) -> str:
    report = state["planning_report"]
    analysis = state["analysis"]
    decision = state["human_decision"]
    prediction = state["prediction"]["prediction"]
    location = state["prediction"]["location"]

    lines = [
        "# ClimateAdapt AI planning review",
        "",
        f"**Workflow run:** `{state['run_id']}`  ",
        f"**Location:** {state['location_name']}  ",
        f"**Planning status:** {report.get('status')}  ",
        f"**Human decision:** {decision.get('action')}  ",
        f"**Reviewer:** {decision.get('reviewer', 'not recorded')}  ",
        f"**Reviewed at:** {decision.get('reviewed_at_utc', 'not recorded')}",
        "",
        "## Verified prediction",
        "",
        f"- Probability: {float(prediction['probability']):.4f}",
        f"- Frozen operational threshold: {float(prediction['operational_threshold']):.2f}",
        f"- Susceptible: {bool(prediction['susceptible'])}",
        f"- Risk band: {prediction['risk_band']}",
        (
            "- Normalised predictive uncertainty: "
            f"{prediction.get('normalised_predictive_uncertainty')}"
        ),
        f"- NZTM2000 coordinate: {location['nztm2000']}",
        f"- WGS84 coordinate: {location['wgs84']}",
        "",
        f"## {report.get('headline', 'Planning brief')}",
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
            "## Approval boundary",
            "",
            "This issue records a human-approved planning-screening artefact. It does not authorise an emergency, engineering, funding, regulatory or land-use action.",
            "",
            f"Reviewer notes: {decision.get('notes', '')}",
            "",
            f"Analysis type: {analysis.get('analysis_type')}",
        ]
    )
    return "\n".join(lines) + "\n"


def gitea_publish_node(state: GiteaWorkflowState) -> dict[str, Any]:
    agent = "gitea_action_agent"
    if state.get("human_decision", {}).get("action") != "approve":
        message = "Gitea publication is prohibited without recorded human approval."
        return {
            "current_agent": agent,
            "status": "gitea_publish_blocked",
            "errors": [message],
            "audit_log": [audit(agent, "create_issue", message)],
        }

    title = f"[ClimateAdapt] Review – {state['location_name']} – {state['run_id']}"
    body = build_issue_body(state)
    try:
        issue = gitea_request(
            state,
            f"{repository_path(state)}/issues",
            method="POST",
            payload={"title": title, "body": body},
        )
        action = {
            "status": "created",
            "repository": f"{state['gitea_owner']}/{state['gitea_repo']}",
            "issue_number": int(issue["number"]),
            "issue_id": int(issue["id"]),
            "title": str(issue["title"]),
            "html_url": str(issue["html_url"]),
            "state": str(issue["state"]),
            "created_at": issue.get("created_at"),
            "expected_body_sha256": __import__("hashlib").sha256(
                body.encode("utf-8")
            ).hexdigest(),
        }
        return {
            "current_agent": agent,
            "status": "gitea_issue_created",
            "gitea_action": action,
            "errors": [],
            "audit_log": [
                audit(agent, "create_issue", f"created issue #{action['issue_number']}")
            ],
        }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return {
            "current_agent": agent,
            "status": "gitea_publish_failed",
            "errors": [message],
            "audit_log": [audit(agent, "create_issue", message)],
        }


def route_after_gitea_publish(state: GiteaWorkflowState) -> str:
    return "finalise" if state.get("errors") else "gitea_verify"


def gitea_verify_node(state: GiteaWorkflowState) -> dict[str, Any]:
    agent = "gitea_verification_agent"
    action = state["gitea_action"]
    number = int(action["issue_number"])
    try:
        # This is a separate read from persistent Gitea state. Verification does
        # not trust the response returned by the publication node.
        issue = gitea_request(
            state,
            f"{repository_path(state)}/issues/{number}",
            method="GET",
        )
        repository = gitea_request(state, repository_path(state), method="GET")
        body = str(issue.get("body", ""))
        observed_hash = __import__("hashlib").sha256(body.encode("utf-8")).hexdigest()
        checks = {
            "issue_number_matches": int(issue.get("number", -1)) == number,
            "title_matches": str(issue.get("title", "")) == action["title"],
            "body_hash_matches": observed_hash == action["expected_body_sha256"],
            "run_id_present": state["run_id"] in body,
            "human_approval_present": "**Human decision:** approve" in body,
            "repository_matches": repository.get("full_name")
            == f"{state['gitea_owner']}/{state['gitea_repo']}",
            "issue_is_open": issue.get("state") == "open",
        }
        passed = all(checks.values())
        verification = {
            "status": "passed" if passed else "failed",
            "verified_at_utc": utc_now(),
            "method": "independent GET of persisted Gitea issue and repository",
            "issue_number": number,
            "html_url": issue.get("html_url"),
            "checks": checks,
            "observed": {
                "repository": repository.get("full_name"),
                "title": issue.get("title"),
                "state": issue.get("state"),
                "created_by": issue.get("user", {}).get("login"),
                "body_sha256": observed_hash,
            },
        }
        if not passed:
            failed = [name for name, value in checks.items() if not value]
            message = "Independent Gitea verification failed: " + ", ".join(failed)
            return {
                "current_agent": agent,
                "status": "gitea_verification_failed",
                "gitea_verification": verification,
                "errors": [message],
                "audit_log": [audit(agent, "verify_issue", message)],
            }
        return {
            "current_agent": agent,
            "status": "gitea_verified",
            "gitea_verification": verification,
            "errors": [],
            "audit_log": [audit(agent, "verify_issue", "passed")],
        }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return {
            "current_agent": agent,
            "status": "gitea_verification_failed",
            "gitea_verification": {"status": "failed", "reason": message},
            "errors": [message],
            "audit_log": [audit(agent, "verify_issue", message)],
        }


def finalise_gitea_node(state: GiteaWorkflowState) -> dict[str, Any]:
    root = Path(state["project_root"])
    output_dir = root / "outputs" / "gitea_agent_workflow_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = state["run_id"]

    if state.get("errors"):
        final_status = "failed"
    elif state.get("human_decision", {}).get("action") == "reject":
        final_status = "human_rejected_no_gitea_action"
    elif state.get("gitea_verification", {}).get("status") == "passed":
        final_status = "human_approved_gitea_verified"
    else:
        final_status = state.get("status", "completed")

    evidence_path = output_dir / f"gitea_evidence_{run_id}.json"
    state_path = output_dir / f"workflow_{run_id}.json"
    brief_path = output_dir / f"planning_brief_{run_id}.md"
    artifacts = {
        "workflow_state": str(state_path.relative_to(root)),
        "gitea_evidence": str(evidence_path.relative_to(root)),
        "planning_brief": str(brief_path.relative_to(root)),
    }

    evidence = {
        "run_id": run_id,
        "final_status": final_status,
        "human_decision": state.get("human_decision", {}),
        "gitea_action": state.get("gitea_action", {}),
        "independent_verification": state.get("gitea_verification", {}),
        "important_note": (
            "No API token is stored in this file. The verification node independently "
            "read the persisted issue from Gitea."
        ),
    }
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    report = state.get("planning_report", {})
    lines = [
        "# ClimateAdapt AI – Approved Gitea Planning Brief",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Location:** {state.get('location_name')}  ",
        f"**Final status:** {final_status}  ",
        f"**Gitea issue:** {state.get('gitea_action', {}).get('html_url', 'not created')}  ",
        f"**Independent verification:** {state.get('gitea_verification', {}).get('status', 'not performed')}",
        "",
        f"## {report.get('headline', 'Planning brief')}",
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
    brief_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    snapshot = dict(state)
    snapshot.update(
        {
            "status": final_status,
            "current_agent": "langgraph_orchestrator",
            "completed_at_utc": utc_now(),
            "artifacts": artifacts,
        }
    )
    # The token is only read from the environment and can never enter snapshot.
    state_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return {
        "current_agent": "langgraph_orchestrator",
        "status": final_status,
        "artifacts": artifacts,
        "audit_log": [audit("langgraph_orchestrator", "workflow_end", final_status)],
    }


def build_gitea_place_graph():
    builder = StateGraph(GiteaWorkflowState)
    builder.add_node("data_validation", data_validation_node)
    builder.add_node("flood_analysis", flood_analysis_node)
    builder.add_node("planning_support", planning_support_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("revision_control", revision_control_node)
    builder.add_node("gitea_publish", gitea_publish_node)
    builder.add_node("gitea_verify", gitea_verify_node)
    builder.add_node("finalise", finalise_gitea_node)

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
        route_after_human_review_for_gitea,
        {
            "gitea_publish": "gitea_publish",
            "revision_control": "revision_control",
            "finalise": "finalise",
        },
    )
    builder.add_conditional_edges(
        "revision_control",
        route_after_revision_control,
        {"planning_support": "planning_support", "finalise": "finalise"},
    )
    builder.add_conditional_edges(
        "gitea_publish",
        route_after_gitea_publish,
        {"gitea_verify": "gitea_verify", "finalise": "finalise"},
    )
    builder.add_edge("gitea_verify", "finalise")
    builder.add_edge("finalise", END)
    return builder.compile(checkpointer=InMemorySaver())
