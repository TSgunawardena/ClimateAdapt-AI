"""Run the live ClimateAdapt three-agent LangGraph workflow for one location."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langgraph.types import Command  # noqa: E402
from climateadapt.agent.live_place_workflow_v1 import (  # noqa: E402
    build_live_place_graph,
    new_run_id,
    utc_now,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    coordinates = parser.add_mutually_exclusive_group(required=True)
    coordinates.add_argument(
        "--latlon",
        nargs=2,
        type=float,
        metavar=("LATITUDE", "LONGITUDE"),
        help="WGS84 coordinate pair.",
    )
    coordinates.add_argument(
        "--nztm",
        nargs=2,
        type=float,
        metavar=("EASTING", "NORTHING"),
        help="NZTM2000 coordinate pair.",
    )
    parser.add_argument("--name", default="Selected Nelson location")
    parser.add_argument("--reviewer", default="Prototype reviewer")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve the review checkpoint for a repeatable demonstration.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="URL of Script 30's running FastAPI service.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--model",
        default="climateadapt-planner:4b",
        help="Fine-tuned Ollama planning model.",
    )
    return parser.parse_args()


def interrupt_value(result: dict):
    interrupts = result.get("__interrupt__", ())
    if not interrupts:
        return None
    return getattr(interrupts[0], "value", interrupts[0])


def main() -> None:
    arguments = parse_arguments()
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

    graph = build_live_place_graph()
    config = {"configurable": {"thread_id": run_id}}
    initial_state = {
        "project_root": str(PROJECT_ROOT),
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "location_name": arguments.name,
        "location_request": request,
        "api_url": arguments.api_url.rstrip("/"),
        "ollama_url": arguments.ollama_url.rstrip("/"),
        "ollama_model": arguments.model,
        "status": "created",
        "current_agent": "langgraph_orchestrator",
        "warnings": [],
        "errors": [],
        "audit_log": [],
        "artifacts": {},
        "revision_count": 0,
    }

    print("\n" + "=" * 78)
    print("CLIMATEADAPT AI – LIVE THREE-AGENT LANGGRAPH WORKFLOW")
    print("=" * 78)
    print(f"Run ID:        {run_id}")
    print(f"Location:      {arguments.name}")
    print(f"Prediction API:{arguments.api_url}")
    print(f"Planning LLM:  {arguments.model}")
    print("Flow: validation → flood analysis → LLM planning → human review")

    result = graph.invoke(initial_state, config=config)
    while True:
        payload = interrupt_value(result)
        if payload is None:
            break

        print("\n" + "-" * 78)
        print("HUMAN-REVIEW CHECKPOINT")
        print("-" * 78)
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        if arguments.auto_approve:
            action = "approve"
            notes = "Approved using --auto-approve for prototype demonstration."
        else:
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
    print("LIVE WORKFLOW COMPLETED")
    print("=" * 78)
    print(f"Final status: {result.get('status')}")
    print(f"Warnings:     {len(result.get('warnings', []))}")
    print(f"Errors:       {len(result.get('errors', []))}")
    for warning in result.get("warnings", []):
        print(f"  WARNING: {warning}")
    for error in result.get("errors", []):
        print(f"  ERROR: {error}")
    print("Artifacts:")
    for name, path in result.get("artifacts", {}).items():
        print(f"  {name}: {path}")
    print("=" * 78)

    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
