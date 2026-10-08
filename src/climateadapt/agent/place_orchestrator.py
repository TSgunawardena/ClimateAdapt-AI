"""Auditable three-agent workflow for an on-demand place analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .state import WorkflowState


class PlaceDataValidationAgent:
    name = "place_data_validation_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        state.current_agent = self.name
        state.status = "validating_place_data"
        result = state.facts.get("place_result", {})
        if result.get("status") != "analysis_ready":
            state.fail(self.name, result.get("reason", "Place analysis is unavailable."))
            return state

        safety_checks = {
            "flood_target_raster_accessed": result.get("flood_target_raster_accessed"),
            "model_retrained": result.get("model_retrained"),
            "threshold_modified": result.get("threshold_modified"),
        }
        if any(value is not False for value in safety_checks.values()):
            state.fail(self.name, f"Place-analysis provenance check failed: {safety_checks}")
            return state

        threshold_path = (
            state.root / "outputs" / "validation_v2" / "selected_threshold_v2.json"
        )
        if not threshold_path.exists():
            state.fail(self.name, "The frozen validation threshold record is missing.")
            return state
        threshold_record = json.loads(threshold_path.read_text(encoding="utf-8"))
        frozen_threshold = float(
            threshold_record.get(
                "operational_threshold", threshold_record.get("selected_threshold")
            )
        )
        if not np.isclose(
            float(result["operational_threshold"]), frozen_threshold, atol=1e-8
        ):
            state.fail(self.name, "Place result does not use the frozen threshold.")
            return state

        coverage = float(result["search_circle_coverage_percentage"])
        if coverage < 20.0:
            state.warnings.append(
                f"Only {coverage:.2f}% of the circular search area contains evaluated land."
            )

        state.facts["data_validation"] = {
            "status": "passed",
            "crs": result.get("crs"),
            "frozen_operational_threshold": frozen_threshold,
            "provenance_checks": safety_checks,
            "search_circle_coverage_percentage": coverage,
        }
        state.status = "data_validated"
        state.log(self.name, "validate_place_result", "passed")
        return state


class PlaceFloodAnalysisAgent:
    name = "place_flood_analysis_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        state.current_agent = self.name
        if state.status != "data_validated":
            state.fail(self.name, "Place data validation must pass first.")
            return state

        result = state.facts["place_result"]
        analysis = {
            "analysis_type": "place-based label-blind susceptibility screening",
            "accuracy_claimed": False,
            "area_name": result["place"]["display_name"],
            "radius_metres": result["radius_metres"],
            "operational_threshold": result["operational_threshold"],
            "evaluated_cells": result["evaluated_land_cells"],
            "evaluated_area_square_kilometres": result[
                "evaluated_land_area_square_kilometres"
            ],
            "susceptible_cells": result["susceptible_cells"],
            "susceptible_area_square_kilometres": result[
                "susceptible_area_square_kilometres"
            ],
            "susceptible_percentage": result[
                "susceptible_percentage_of_evaluated_land"
            ],
            "mean_probability": result["mean_probability"],
            "maximum_probability": result["maximum_probability"],
            "mean_uncertainty_indicator": result["mean_uncertainty"],
            "high_uncertainty_percentage": result["high_uncertainty_percentage"],
            "median_elevation_metres": result["median_elevation_metres"],
            "median_slope_degrees": result["median_slope_degrees"],
            "median_distance_to_coast_metres": result[
                "median_distance_to_coast_metres"
            ],
            "risk_band_cells": result["risk_band_cells"],
            "leading_susceptible_land_cover_codes": result[
                "leading_susceptible_land_cover_codes"
            ],
        }
        state.facts["flood_analysis"] = analysis
        state.status = "susceptibility_analysed"
        state.log(self.name, "analyse_place_outputs", "completed")
        return state


class PlacePlanningSupportAgent:
    name = "place_planning_support_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        state.current_agent = self.name
        if state.status != "susceptibility_analysed":
            state.fail(self.name, "Place susceptibility analysis must complete first.")
            return state

        analysis = state.facts["flood_analysis"]
        recommendations = [
            {
                "priority": "1",
                "action": "Review mapped susceptible locations",
                "reason": (
                    f"Within the {analysis['radius_metres']:,} m search radius, "
                    f"{analysis['susceptible_area_square_kilometres']:.3f} km² "
                    f"({analysis['susceptible_percentage']:.2f}% of evaluated land) "
                    "meets the frozen operational threshold."
                ),
            },
            {
                "priority": "2",
                "action": "Inspect uncertain locations before prioritisation",
                "reason": (
                    f"{analysis['high_uncertainty_percentage']:.2f}% of evaluated "
                    "cells have entropy uncertainty of at least 0.80."
                ),
            },
            {
                "priority": "3",
                "action": "Request site-specific technical evidence",
                "reason": (
                    "The model does not simulate water depth, velocity, tides, "
                    "storm surge, rainfall, drainage capacity or asset exposure."
                ),
            },
            {
                "priority": "4",
                "action": "Keep a qualified human reviewer in control",
                "reason": (
                    "The result may support further investigation but cannot "
                    "authorise engineering, regulatory or emergency action."
                ),
            },
        ]
        state.recommendations = recommendations

        place_result = state.facts["place_result"]
        run_directory = state.root / Path(
            place_result["artifacts"]["analysis_json"]
        ).parent
        report_json = run_directory / "place_agent_report.json"
        report_markdown = run_directory / "place_planning_brief.md"
        state_path = run_directory / "place_workflow_state.json"

        report_payload = {
            "run_id": state.run_id,
            "area": state.area_name,
            "status": "human_review_required",
            "data_validation": state.facts["data_validation"],
            "flood_analysis": analysis,
            "planning_recommendations": recommendations,
            "mandatory_limitations": [
                "Screening-level susceptibility; no local accuracy claim.",
                "Not a hydrodynamic flood simulation.",
                "Not an emergency-warning, engineering or regulatory product.",
                "Requires GIS, engineering, community and human review.",
            ],
        }
        report_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        lines = [
            f"# ClimateAdapt AI – {state.area_name} Planning Brief",
            "",
            f"**Workflow run:** `{state.run_id}`  ",
            "**Decision status:** Human review required",
            "",
            "## Verified screening output",
            "",
            f"- Search radius: {analysis['radius_metres']:,} m",
            f"- Evaluated land area: {analysis['evaluated_area_square_kilometres']:.3f} km²",
            f"- Operational threshold: {analysis['operational_threshold']:.2f}",
            f"- Susceptible area: {analysis['susceptible_area_square_kilometres']:.3f} km²",
            f"- Susceptible proportion: {analysis['susceptible_percentage']:.2f}%",
            f"- High-uncertainty proportion: {analysis['high_uncertainty_percentage']:.2f}%",
            "",
            "## Planning observations",
            "",
        ]
        for item in recommendations:
            lines.append(f"{item['priority']}. **{item['action']}** – {item['reason']}")
        lines.extend(
            [
                "",
                "## Mandatory limitations",
                "",
                "- This output is for preliminary planning screening only.",
                "- It does not estimate flood depth, velocity or event timing.",
                "- It must not be used alone for emergency or regulatory decisions.",
                "- Final decisions require qualified human and community review.",
            ]
        )
        report_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

        state.artifacts.update(place_result["artifacts"])
        state.artifacts["agent_report_json"] = str(report_json.relative_to(state.root))
        state.artifacts["agent_report_markdown"] = str(
            report_markdown.relative_to(state.root)
        )
        state.artifacts["workflow_state"] = str(state_path.relative_to(state.root))
        state.status = "human_review_required"
        state.log(self.name, "prepare_place_planning_brief", "completed")
        state.save(state_path)
        return state


class PlaceAnalysisOrchestrator:
    """Run three deterministic agents for one precomputed place result."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.agents = [
            PlaceDataValidationAgent(),
            PlaceFloodAnalysisAgent(),
            PlacePlanningSupportAgent(),
        ]

    def run(self, place_result: dict) -> WorkflowState:
        state = WorkflowState(
            project_root=str(self.project_root),
            area_name=place_result["place"]["display_name"],
        )
        state.run_id = place_result["run_id"]
        state.facts["place_result"] = place_result
        state.log("place_orchestrator", "workflow_start", "started")

        for agent in self.agents:
            try:
                state = agent.run(state)
            except Exception as error:
                state.fail(agent.name, f"{type(error).__name__}: {error}")
            if state.status == "failed":
                break

        if state.status != "failed":
            state.log("place_orchestrator", "workflow_end", "human review required")
            workflow_path = state.root / state.artifacts["workflow_state"]
            state.save(workflow_path)
        return state
