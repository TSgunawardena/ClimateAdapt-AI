"""LangGraph node 1: validate completed multi-paradigm model evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .state_v2 import (
    ClimateAdaptGraphState,
    audit_event,
    find_number,
    load_json,
)


AGENT_NAME = "data_validation_agent_v2"


def data_validation_node(state: ClimateAdaptGraphState) -> dict:
    root = Path(state["project_root"])
    relative_paths = {
        "test_metrics": "outputs/test_v2/tahunanui_test_metrics_v2.json",
        "baseline_comparison": "outputs/baselines_v2/baseline_comparison_v2.json",
        "uncertainty_summary": "outputs/uncertainty_v2/tahunanui_mc_dropout_summary_v2.json",
        "physicalist_summary": "outputs/physicalist_guardrail_v2/tahunanui_physicalist_guardrail_summary_v2.json",
        "threshold_record": "outputs/validation_v2/selected_threshold_v2.json",
        "mc_mean_probability": "outputs/uncertainty_v2/tahunanui_mc_mean_probability_v2.tif",
        "physicalist_decision": "outputs/physicalist_guardrail_v2/tahunanui_physicalist_decision_v2.tif",
        "checkpoint": "models/climateadapt_unet_v2_best.pt",
    }
    paths = {name: root / relative for name, relative in relative_paths.items()}
    missing = [relative_paths[name] for name, path in paths.items() if not path.exists()]
    if missing:
        message = "Missing required artifacts: " + ", ".join(missing)
        return {
            "current_agent": AGENT_NAME,
            "status": "validation_failed",
            "paths": relative_paths,
            "errors": [message],
            "audit_log": [audit_event(AGENT_NAME, "validate_artifacts", message)],
        }

    try:
        test = load_json(paths["test_metrics"])
        baseline = load_json(paths["baseline_comparison"])
        uncertainty = load_json(paths["uncertainty_summary"])
        physicalist = load_json(paths["physicalist_summary"])
        threshold_record = load_json(paths["threshold_record"])

        frozen_threshold = find_number(
            threshold_record,
            ("operational_threshold", "safety_threshold", "selected_threshold"),
        )
        if frozen_threshold is None:
            raise ValueError("No frozen operational threshold was found.")

        threshold_sources = {
            "validation": frozen_threshold,
            "test": find_number(
                test,
                ("threshold", "operational_threshold", "frozen_operational_threshold"),
            ),
            "uncertainty": find_number(
                uncertainty,
                ("frozen_operational_threshold", "operational_threshold"),
            ),
            "physicalist": find_number(physicalist, ("operational_threshold",)),
        }
        mismatches = {
            name: value
            for name, value in threshold_sources.items()
            if value is None or not np.isclose(value, frozen_threshold, atol=1e-8)
        }
        if mismatches:
            raise ValueError(f"Frozen threshold mismatch or omission: {mismatches}")

        if test.get("threshold_modified_using_test_data") is not False:
            raise ValueError("Test provenance does not confirm a frozen threshold.")
        if uncertainty.get("threshold_modified_using_test_data") is not False:
            raise ValueError("MC-dropout provenance does not confirm a frozen threshold.")
        if uncertainty.get("model_weights_modified") is not False:
            raise ValueError("MC-dropout output reports modified model weights.")
        if physicalist.get("label_blind_test_application") is not True:
            raise ValueError("Physicalist output is not marked as label-blind application.")

        raster_metadata = {}
        reference_signature = None
        for name in ("mc_mean_probability", "physicalist_decision"):
            with rasterio.open(paths[name]) as source:
                signature = (
                    source.width,
                    source.height,
                    source.crs.to_string() if source.crs else None,
                    tuple(source.transform),
                )
                raster_metadata[name] = {
                    "width": source.width,
                    "height": source.height,
                    "crs": signature[2],
                    "resolution": [abs(source.transform.a), abs(source.transform.e)],
                    "nodata": source.nodata,
                }
                if reference_signature is None:
                    reference_signature = signature
                elif signature != reference_signature:
                    raise ValueError(f"Raster alignment mismatch for {name}.")
        if reference_signature[2] != "EPSG:2193":
            raise ValueError("Expected NZTM2000 / EPSG:2193 output rasters.")

        validation_summary = {
            "status": "passed",
            "frozen_operational_threshold": frozen_threshold,
            "threshold_sources": threshold_sources,
            "raster_metadata": raster_metadata,
            "raster_alignment": "passed",
            "crs_check": "passed",
            "provenance_check": "passed",
            "json_artifacts_loaded": [
                "test_metrics",
                "baseline_comparison",
                "uncertainty_summary",
                "physicalist_summary",
            ],
        }
        return {
            "current_agent": AGENT_NAME,
            "status": "data_validated",
            "paths": relative_paths,
            "validation_summary": validation_summary,
            "errors": [],
            "warnings": [],
            "audit_log": [audit_event(AGENT_NAME, "validate_artifacts", "passed")],
        }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return {
            "current_agent": AGENT_NAME,
            "status": "validation_failed",
            "paths": relative_paths,
            "errors": [message],
            "audit_log": [audit_event(AGENT_NAME, "validate_artifacts", message)],
        }

