"""Run the ClimateAdapt LangGraph V2 workflow with human approval."""

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
from climateadapt.agent.langgraph_workflow_v2 import (  # noqa: E402
    build_climateadapt_graph,
)
from climateadapt.agent.state_v2 import new_run_id, utc_now  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve the human checkpoint for a repeatable automated demo.",
    )
    parser.add_argument(
        "--reviewer",
        default="Prototype reviewer",
        help="Reviewer name recorded in the audit output.",
    )
    return parser.parse_args()


def get_interrupt_payload(result: dict):
    interrupts = result.get("__interrupt__", ())
    if not interrupts:
        return None
    return getattr(interrupts[0], "value", interrupts[0])


def main() -> None:
    arguments = parse_arguments()
    run_id = new_run_id()
    graph = build_climateadapt_graph()
    config = {"configurable": {"thread_id": run_id}}
    initial_state = {
        "project_root": str(PROJECT_ROOT),
        "area_name": "Tāhunanui independent test area",
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "status": "created",
        "current_agent": "langgraph_orchestrator",
        "warnings": [],
        "errors": [],
        "audit_log": [],
        "artifacts": {},
        "recommendations": [],
        "revision_count": 0,
    }

    print("\n" + "=" * 78)
    print("CLIMATEADAPT AI – LANGGRAPH THREE-AGENT WORKFLOW V2")
    print("=" * 78)
    print(f"Run ID: {run_id}")
    print("Starting graph: validation → analysis → planning → conditional review")

    result = graph.invoke(initial_state, config=config)
    while True:
        payload = get_interrupt_payload(result)
        if payload is None:
            break
        print("\n" + "-" * 78)
        print("LANGGRAPH HUMAN-REVIEW INTERRUPT")
        print("-" * 78)
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        if arguments.auto_approve:
            action = "approve"
            notes = "Approved through --auto-approve for prototype demonstration."
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
    print("LANGGRAPH WORKFLOW RESULT")
    print("=" * 78)
    print(f"Final status: {result.get('status')}")
    print(f"Warnings: {len(result.get('warnings', []))}")
    print(f"Errors: {len(result.get('errors', []))}")
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

