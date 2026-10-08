"""Shared typed state and utilities for the ClimateAdapt LangGraph workflow."""

from __future__ import annotations

import json
import operator
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from typing_extensions import TypedDict


class ClimateAdaptGraphState(TypedDict, total=False):
    project_root: str
    area_name: str
    run_id: str
    started_at_utc: str
    status: str
    current_agent: str
    paths: dict[str, str]
    validation_summary: dict[str, Any]
    flood_analysis: dict[str, Any]
    planning_summary: dict[str, Any]
    recommendations: list[dict[str, str]]
    human_review_required: bool
    human_decision: dict[str, str]
    revision_count: int
    artifacts: dict[str, str]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    audit_log: Annotated[list[dict[str, str]], operator.add]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def audit_event(agent: str, action: str, outcome: str) -> dict[str, str]:
    return {
        "timestamp_utc": utc_now(),
        "agent": agent,
        "action": action,
        "outcome": outcome,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def find_number(value: Any, keys: tuple[str, ...]) -> float | None:
    """Find the first numeric value whose key matches one of the supplied keys."""
    normalised = {key.lower() for key in keys}
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in normalised and isinstance(nested, (int, float)):
                return float(nested)
        for nested in value.values():
            found = find_number(nested, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_number(nested, keys)
            if found is not None:
                return found
    return None


def find_mapping(value: Any, key_fragment: str) -> dict[str, Any] | None:
    """Locate a nested mapping identified by a descriptive key fragment."""
    fragment = key_fragment.lower()
    if isinstance(value, dict):
        for key, nested in value.items():
            if fragment in key.lower() and isinstance(nested, dict):
                return nested
        for nested in value.values():
            found = find_mapping(nested, key_fragment)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_mapping(nested, key_fragment)
            if found is not None:
                return found
    return None

