"""Agent 3: convert verified facts into guarded planning observations."""

from __future__ import annotations

import json

from .state import WorkflowState


class PlanningSupportAgent:
    name = "planning_support_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        state.current_agent = self.name
        if state.status != "susceptibility_analysed":
            state.fail(self.name, "Flood analysis must complete before planning support.")
            return state

        state.status = "preparing_planning_support"
        analysis = state.facts["flood_analysis"]
        threshold = analysis["operational_threshold"]
        susceptible_percentage = analysis["susceptible_percentage"]
        susceptible_area = analysis["susceptible_area_square_kilometres"]
        high_uncertainty_percentage = analysis["high_uncertainty_percentage"]

        recommendations = [
            {
                "priority": "1",
                "action": "Review operational susceptibility locations",
                "reason": (
                    f"The frozen {threshold:.2f} threshold identifies "
                    f"{susceptible_area:.3f} km² ({susceptible_percentage:.2f}%) "
                    "for planning-level review."
                ),
            },
            {
                "priority": "2",
                "action": "Request engineering assessment before intervention",
                "reason": (
                    "The model is a screening tool and does not simulate flood "
                    "depth, velocity, drainage capacity, tides, or storm surge."
                ),
            },
            {
                "priority": "3",
                "action": "Inspect high-uncertainty locations",
                "reason": (
                    f"{high_uncertainty_percentage:.2f}% of evaluated cells have "
                    "normalised entropy of at least 0.80."
                ),
            },
            {
                "priority": "4",
                "action": "Keep a human decision-maker in control",
                "reason": (
                    "No automated land-use, emergency, funding, cultural-value, "
                    "or public-safety decision should be made from this output."
                ),
            },
        ]
        state.recommendations = recommendations

        output_dir = state.root / "outputs" / "agent_workflow_v1"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_json = output_dir / "waimea_monaco_agent_report.json"
        report_markdown = output_dir / "waimea_monaco_agent_report.md"

        report_payload = {
            "run_id": state.run_id,
            "area": state.area_name,
            "status": "human_review_required",
            "data_validation": state.facts["data_validation"],
            "flood_analysis": analysis,
            "planning_recommendations": recommendations,
            "mandatory_limitations": [
                "Label-blind demonstration; no accuracy claim.",
                "Not a hydrodynamic flood simulation.",
                "Not an emergency-warning or regulatory product.",
                "Requires GIS, engineering, community and human review.",
            ],
        }
        with report_json.open("w", encoding="utf-8") as file:
            json.dump(report_payload, file, indent=2)

        lines = [
            "# ClimateAdapt AI – Waimea Inlet / Monaco Planning Brief",
            "",
            f"**Workflow run:** `{state.run_id}`  ",
            "**Decision status:** Human review required",
            "",
            "## Verified model output",
            "",
            f"- Evaluated area: {analysis['evaluated_area_square_kilometres']:.3f} km²",
            f"- Operational threshold: {threshold:.2f}",
            f"- Predicted susceptible area: {susceptible_area:.3f} km²",
            f"- Predicted susceptible proportion: {susceptible_percentage:.2f}%",
            f"- High-uncertainty proportion: {high_uncertainty_percentage:.2f}%",
            "",
            "## Planning observations",
            "",
        ]
        for item in recommendations:
            lines.append(
                f"{item['priority']}. **{item['action']}** – {item['reason']}"
            )
        lines.extend(
            [
                "",
                "## Mandatory limitations",
                "",
                "- This is a label-blind susceptibility demonstration.",
                "- It does not estimate water depth, velocity or event timing.",
                "- It must not be used alone for emergency or regulatory decisions.",
                "- Final decisions require qualified human and community review.",
            ]
        )
        report_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

        state.artifacts["agent_report_json"] = str(report_json.relative_to(state.root))
        state.artifacts["agent_report_markdown"] = str(
            report_markdown.relative_to(state.root)
        )
        state.status = "human_review_required"
        state.log(self.name, "prepare_planning_brief", "completed")
        return state
