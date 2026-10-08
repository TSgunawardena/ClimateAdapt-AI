"""Sequential, auditable orchestration for the three ClimateAdapt agents."""

from __future__ import annotations

from pathlib import Path

from .data_validation_agent import DataValidationAgent
from .flood_analysis_agent import FloodAnalysisAgent
from .planning_support_agent import PlanningSupportAgent
from .state import WorkflowState


class ClimateAdaptOrchestrator:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.output_dir = self.project_root / "outputs" / "agent_workflow_v1"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agents = [
            DataValidationAgent(),
            FloodAnalysisAgent(),
            PlanningSupportAgent(),
        ]

    def run(self) -> WorkflowState:
        state = WorkflowState(project_root=str(self.project_root))
        state.log("orchestrator", "workflow_start", "started")
        state_path = self.output_dir / "workflow_state.json"

        for agent in self.agents:
            print(f"\n[{agent.name}] starting...")
            try:
                state = agent.run(state)
            except Exception as error:
                state.fail(agent.name, f"{type(error).__name__}: {error}")
            state.save(state_path)
            print(f"[{agent.name}] status: {state.status}")
            if state.status == "failed":
                break

        if state.status != "failed":
            state.log("orchestrator", "workflow_end", "human review required")
            state.artifacts["workflow_state"] = str(state_path.relative_to(state.root))
            state.save(state_path)

        return state
