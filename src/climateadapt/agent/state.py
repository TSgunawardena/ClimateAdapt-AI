"""Shared state passed between ClimateAdapt AI agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid


@dataclass
class WorkflowState:
    project_root: str
    area_name: str = "Waimea inlet and Monaco"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "created"
    current_agent: str = "orchestrator"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    recommendations: list[dict[str, str]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    audit_log: list[dict[str, str]] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return Path(self.project_root)

    def log(self, agent: str, action: str, outcome: str) -> None:
        self.audit_log.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "agent": agent,
                "action": action,
                "outcome": outcome,
            }
        )

    def fail(self, agent: str, message: str) -> None:
        self.status = "failed"
        self.current_agent = agent
        self.errors.append(message)
        self.log(agent, "workflow_stop", message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)
