"""LangGraph node 2: combine Connectionist, Bayesian and Physicalist evidence."""

from __future__ import annotations

from pathlib import Path

from .state_v2 import (
    ClimateAdaptGraphState,
    audit_event,
    find_mapping,
    find_number,
    load_json,
)


AGENT_NAME = "flood_analysis_agent_v2"


def _metric(mapping: dict | None, *keys: str) -> float | None:
    if not mapping:
        return None
    return find_number(mapping, tuple(keys))


def flood_analysis_node(state: ClimateAdaptGraphState) -> dict:
    if state.get("status") != "data_validated":
        message = "Data validation must pass before multi-paradigm analysis."
        return {
            "current_agent": AGENT_NAME,
            "status": "analysis_failed",
            "errors": [message],
            "audit_log": [audit_event(AGENT_NAME, "combine_evidence", message)],
        }

    try:
        root = Path(state["project_root"])
        paths = state["paths"]
        test = load_json(root / paths["test_metrics"])
        baseline = load_json(root / paths["baseline_comparison"])
        uncertainty = load_json(root / paths["uncertainty_summary"])
        physicalist = load_json(root / paths["physicalist_summary"])

        operational = test.get("operational_safety_threshold_metrics", {})
        threshold_free = test.get("threshold_free_metrics", {})
        guardrail = physicalist.get("results", {})
        logistic = find_mapping(baseline, "logistic")
        baseline_unet = find_mapping(baseline, "unet")

        unet_recall = _metric(operational, "recall_sensitivity", "recall")
        logistic_recall = _metric(logistic, "recall_sensitivity", "recall")
        recall_gap = (
            abs(logistic_recall - unet_recall)
            if logistic_recall is not None and unet_recall is not None
            else None
        )
        model_disagreement = recall_gap is not None and recall_gap >= 0.20

        human_review_cells = int(guardrail.get("human_review_cells", 0))
        supported_cells = int(guardrail.get("physically_supported_cells", 0))
        review_percentage = float(
            guardrail.get("human_review_percentage_of_supported", 0.0)
        )
        uncertainty_error_auc = float(
            uncertainty.get("uncertainty_as_error_detector_roc_auc", 0.5)
        )
        uncertainty_enrichment = float(
            uncertainty.get("high_uncertainty_error_enrichment", 1.0)
        )
        review_required = (
            human_review_cells > 0
            or model_disagreement
            or uncertainty_error_auc >= 0.70
        )

        if review_percentage >= 25.0 or model_disagreement:
            planning_priority = "HIGH"
        elif review_required:
            planning_priority = "MODERATE"
        else:
            planning_priority = "ROUTINE"

        analysis = {
            "analysis_type": "audited multi-paradigm evidence synthesis",
            "area": "Tāhunanui independent geographic test area",
            "operational_threshold": state["validation_summary"][
                "frozen_operational_threshold"
            ],
            "connectionist_unet": {
                "balanced_accuracy": _metric(operational, "balanced_accuracy"),
                "precision": _metric(operational, "precision"),
                "recall": unet_recall,
                "f2_score": _metric(operational, "f2_score"),
                "dice": _metric(operational, "dice"),
                "iou": _metric(operational, "iou"),
                "roc_auc": _metric(threshold_free, "roc_auc"),
                "average_precision": _metric(
                    threshold_free, "average_precision_pr_auc", "average_precision"
                ),
                "brier_score": _metric(threshold_free, "brier_score"),
            },
            "statistical_logistic_baseline": {
                "balanced_accuracy": _metric(logistic, "balanced_accuracy"),
                "precision": _metric(logistic, "precision"),
                "recall": logistic_recall,
                "f2_score": _metric(logistic, "f2_score"),
                "dice": _metric(logistic, "dice"),
                "roc_auc": _metric(logistic, "roc_auc"),
                "average_precision": _metric(
                    logistic, "average_precision", "average_precision_pr_auc"
                ),
                "comparison_record_available": logistic is not None,
                "reported_unet_comparison_available": baseline_unet is not None,
            },
            "bayesian_style_uncertainty": {
                "method": uncertainty.get("method", "Monte Carlo Dropout"),
                "mc_passes": uncertainty.get("mc_passes"),
                "mean_epistemic_standard_deviation": uncertainty.get(
                    "mean_epistemic_standard_deviation"
                ),
                "overall_error_rate": uncertainty.get("overall_error_rate"),
                "top_uncertainty_error_rate": uncertainty.get(
                    "high_uncertainty_error_rate"
                ),
                "error_enrichment": uncertainty_enrichment,
                "uncertainty_error_roc_auc": uncertainty_error_auc,
            },
            "physicalist_guardrail": {
                **guardrail,
                "physical_limits": physicalist.get("physical_limits", {}),
            },
            "cross_model_recall_gap": recall_gap,
            "material_model_disagreement": model_disagreement,
            "planning_priority": planning_priority,
            "human_review_required": review_required,
            "scientific_boundary": (
                "Screening evidence only; not a hydrodynamic flood-depth, "
                "velocity, timing, emergency-warning or regulatory model."
            ),
        }
        warnings = []
        if logistic is None:
            warnings.append(
                "The baseline JSON was present, but a logistic-regression section "
                "could not be identified for metric comparison."
            )
        if model_disagreement:
            warnings.append(
                "The logistic baseline and U-Net have a material recall difference; "
                "a human must interpret the precision–recall trade-off."
            )
        return {
            "current_agent": AGENT_NAME,
            "status": "multi_paradigm_analysis_completed",
            "flood_analysis": analysis,
            "human_review_required": review_required,
            "warnings": warnings,
            "errors": [],
            "audit_log": [audit_event(AGENT_NAME, "combine_evidence", "completed")],
        }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return {
            "current_agent": AGENT_NAME,
            "status": "analysis_failed",
            "errors": [message],
            "audit_log": [audit_event(AGENT_NAME, "combine_evidence", message)],
        }

