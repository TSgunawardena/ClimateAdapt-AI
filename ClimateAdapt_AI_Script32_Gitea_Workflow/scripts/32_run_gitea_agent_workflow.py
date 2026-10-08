"""Run ClimateAdapt's live LangGraph workflow with verified Gitea action."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langgraph.types import Command  # noqa: E402
from climateadapt.agent.gitea_workflow_v1 import (  # noqa: E402
    build_gitea_place_graph,
)
from climateadapt.agent.live_place_workflow_v1 import new_run_id, utc_now  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    coordinates = parser.add_mutually_exclusive_group(required=True)
    coordinates.add_argument(
        "--latlon",
        nargs=2,
        type=float,
        metavar=("LATITUDE", "LONGITUDE"),
    )
    coordinates.add_argument(
        "--nztm",
        nargs=2,
        type=float,
        metavar=("EASTING", "NORTHING"),
    )
    parser.add_argument("--name", default="Selected Nelson location")
    parser.add_argument("--reviewer", default="Prototype reviewer")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="climateadapt-planner:4b")
    return parser.parse_args()


def require_gitea_environment() -> dict[str, str]:
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


def get_interrupt_payload(result: dict):
    interrupts = result.get("__interrupt__", ())
    if not interrupts:
        return None
    return getattr(interrupts[0], "value", interrupts[0])


def main() -> None:
    arguments = parse_arguments()
    gitea = require_gitea_environment()
    run_id = new_run_id()
    request = {"location_name": arguments.name}
    if arguments.latlon:
        request.update(
            {"latitude": arguments.latlon[0], "longitude": arguments.latlon[1]}
        )
    else:
        request.update(
            {"easting": arguments.nztm[0], "northing": arguments.nztm[1]}
        )

    initial_state = {
        "project_root": str(PROJECT_ROOT),
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "location_name": arguments.name,
        "location_request": request,
        "api_url": arguments.api_url.rstrip("/"),
        "ollama_url": arguments.ollama_url.rstrip("/"),
        "ollama_model": arguments.model,
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
    config = {"configurable": {"thread_id": run_id}}
    graph = build_gitea_place_graph()

    print("\n" + "=" * 78)
    print("CLIMATEADAPT AI – LANGGRAPH + VERIFIED GITEA WORKFLOW")
    print("=" * 78)
    print(f"Run ID:       {run_id}")
    print(f"Location:     {arguments.name}")
    print(f"Planning LLM: {arguments.model}")
    print(
        "Gitea target: "
        f"{gitea['CLIMATEADAPT_GITEA_OWNER']}/"
        f"{gitea['CLIMATEADAPT_GITEA_REPO']}"
    )
    print("Token:        configured but never logged")
    print(
        "Flow: validation → analysis → LLM planning → human review → "
        "Gitea action → independent verification"
    )

    result = graph.invoke(initial_state, config=config)
    while True:
        payload = get_interrupt_payload(result)
        if payload is None:
            break

        print("\n" + "-" * 78)
        print("HUMAN-REVIEW CHECKPOINT")
        print("-" * 78)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(
            "IMPORTANT: approve will publish a new issue to the configured "
            "private Gitea repository."
        )
        action = input("Action [approve/revise/reject]: ").strip().lower()
        while action not in {"approve", "revise", "reject"}:
            action = input("Please enter approve, revise, or reject: ").strip().lower()
        notes = input("Reviewer notes (optional): ").strip()

        result = graph.invoke(
            Command(
                resume={
                    "action": action,
                    "reviewer": arguments.reviewer,
                    "notes": notes,
                }
            ),
            config=config,
        )

    print("\n" + "=" * 78)
    print("GITEA WORKFLOW COMPLETED")
    print("=" * 78)
    print(f"Final status: {result.get('status')}")
    print(f"Warnings:     {len(result.get('warnings', []))}")
    print(f"Errors:       {len(result.get('errors', []))}")
    for warning in result.get("warnings", []):
        print(f"  WARNING: {warning}")
    for error in result.get("errors", []):
        print(f"  ERROR: {error}")

    action = result.get("gitea_action", {})
    verification = result.get("gitea_verification", {})
    if action:
        print(f"Gitea issue:  #{action.get('issue_number')} {action.get('html_url')}")
    if verification:
        print(f"Verification: {verification.get('status')}")
    print("Artifacts:")
    for name, path in result.get("artifacts", {}).items():
        print(f"  {name}: {path}")
    print("=" * 78)

    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
