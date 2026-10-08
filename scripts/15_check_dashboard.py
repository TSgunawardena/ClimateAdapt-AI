"""Check that the ClimateAdapt dashboard has its required inputs."""

from pathlib import Path
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    PROJECT_ROOT / "dashboard/app.py",
    PROJECT_ROOT / "outputs/agent_workflow_v1/workflow_state.json",
    PROJECT_ROOT / "outputs/agent_workflow_v1/waimea_monaco_agent_report.json",
    PROJECT_ROOT
    / "outputs/demonstration_v2/waimea_monaco_demo_summary_v2.json",
    PROJECT_ROOT / "evidence/figures/waimea_monaco_demonstration_v2.png",
]


def main() -> None:
    print("\nClimateAdapt AI dashboard check")
    print("-" * 45)
    failed = False
    for path in REQUIRED_FILES:
        ready = path.exists()
        print(f"{'READY' if ready else 'MISSING':<8} {path.relative_to(PROJECT_ROOT)}")
        failed |= not ready

    streamlit_ready = importlib.util.find_spec("streamlit") is not None
    print(f"{'READY' if streamlit_ready else 'MISSING':<8} Streamlit package")
    failed |= not streamlit_ready

    if failed:
        raise SystemExit("Dashboard prerequisites are incomplete.")

    print("-" * 45)
    print("Dashboard is ready.")
    print(r"Run: streamlit run .\dashboard\app.py")


if __name__ == "__main__":
    main()
