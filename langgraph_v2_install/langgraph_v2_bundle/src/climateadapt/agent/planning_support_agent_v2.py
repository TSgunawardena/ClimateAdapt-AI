"""LangGraph node 3: produce guarded, evidence-linked planning advice."""

from __future__ import annotations

from .state_v2 import ClimateAdaptGraphState, audit_event


AGENT_NAME = "planning_support_agent_v2"


def planning_support_node(state: ClimateAdaptGraphState) -> dict:
    if state.get("status") not in {
        "multi_paradigm_analysis_completed",
        "human_revise",
    }:
        message = "Multi-paradigm flood analysis must complete first."
        return {
            "current_agent": AGENT_NAME,
            "status": "planning_failed",
            "errors": [message],
            "audit_log": [audit_event(AGENT_NAME, "prepare_guidance", message)],
        }

    analysis = state["flood_analysis"]
    guardrail = analysis["physicalist_guardrail"]
    uncertainty = analysis["bayesian_style_uncertainty"]
    threshold = analysis["operational_threshold"]
    revision_count = int(state.get("revision_count", 0))

    recommendations = [
        {
            "priority": "1",
            "action": "Review physically supported susceptibility corridors",
            "reason": (
                f"The frozen {threshold:.2f} model threshold produced "
                f"{guardrail.get('original_candidate_cells', 0):,} candidate cells; "
                f"{guardrail.get('physically_supported_cells', 0):,} remained after "
                "terrain and coastal-connectivity checks."
            ),
        },
        {
            "priority": "2",
            "action": "Inspect amber human-review locations",
            "reason": (
                f"{guardrail.get('human_review_cells', 0):,} supported cells "
                f"({guardrail.get('human_review_percentage_of_supported', 0.0):.2f}%) "
                "were flagged by epistemic uncertainty or predictive entropy."
            ),
        },
        {
            "priority": "3",
            "action": "Compare conservative and precise model behaviours",
            "reason": (
                "The logistic baseline emphasises sensitivity while the U-Net "
                "provides more spatially precise segmentation; disagreement must "
                "be interpreted rather than automatically averaged."
            ),
        },
        {
            "priority": "4",
            "action": "Obtain qualified site-specific assessment",
            "reason": (
                "The prototype does not simulate water depth, velocity, event "
                "timing, tides, rainfall, drainage capacity or exposed assets."
            ),
        },
    ]
    previous_decision = state.get("human_decision", {})
    if previous_decision.get("action") == "revise":
        revision_count += 1
        recommendations.insert(
            0,
            {
                "priority": "0",
                "action": "Address reviewer feedback",
                "reason": previous_decision.get("notes", "Revision requested."),
            },
        )

    planning_summary = {
        "priority_level": analysis["planning_priority"],
        "decision_status": (
            "human_review_required"
            if analysis["human_review_required"]
            else "automated_screening_complete"
        ),
        "recommendations": recommendations,
        "mandatory_limitations": [
            "Independent test evidence is evaluation evidence, not a local guarantee.",
            "Monte Carlo dropout is approximate epistemic uncertainty, not a full posterior.",
            "The Physicalist component is an explainable screening guardrail, not hydrodynamics.",
            "No emergency, engineering, funding or regulatory decision may be automated.",
        ],
        "reviewer_feedback_incorporated": previous_decision.get("action") == "revise",
        "revision_count": revision_count,
        "uncertainty_error_enrichment": uncertainty.get("error_enrichment"),
    }
    return {
        "current_agent": AGENT_NAME,
        "status": "planning_support_prepared",
        "planning_summary": planning_summary,
        "recommendations": recommendations,
        "revision_count": revision_count,
        "human_review_required": analysis["human_review_required"],
        "errors": [],
        "audit_log": [audit_event(AGENT_NAME, "prepare_guidance", "completed")],
    }

