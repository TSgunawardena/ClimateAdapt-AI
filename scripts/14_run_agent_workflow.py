"""Run the ClimateAdapt AI three-agent prototype workflow."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.agent import ClimateAdaptOrchestrator  # noqa: E402


def main() -> None:
    print("\n" + "=" * 78)
    print("CLIMATEADAPT AI – THREE-AGENT PROTOTYPE")
    print("=" * 78)
    orchestrator = ClimateAdaptOrchestrator(PROJECT_ROOT)
    state = orchestrator.run()

    print("\n" + "=" * 78)
    print("WORKFLOW RESULT")
    print("=" * 78)
    print(f"Run ID: {state.run_id}")
    print(f"Final status: {state.status}")
    print(f"Warnings: {len(state.warnings)}")
    print(f"Errors: {len(state.errors)}")
    for error in state.errors:
        print(f"  ERROR: {error}")
    print("Artifacts:")
    for name, path in state.artifacts.items():
        print(f"  {name}: {path}")
    print("=" * 78)

    if state.status == "failed":
        raise SystemExit(1)
    print("The workflow stopped at the required human-review checkpoint.")


if __name__ == "__main__":
    main()
