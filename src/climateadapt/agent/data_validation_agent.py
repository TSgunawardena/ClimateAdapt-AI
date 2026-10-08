"""Agent 1: verify model outputs and their geospatial compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from .state import WorkflowState


class DataValidationAgent:
    name = "data_validation_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        state.current_agent = self.name
        state.status = "validating_data"
        root = state.root

        paths = {
            "summary": root
            / "outputs/demonstration_v2/waimea_monaco_demo_summary_v2.json",
            "probability": root
            / "outputs/demonstration_v2/waimea_monaco_probability_v2.tif",
            "binary": root
            / "outputs/demonstration_v2/waimea_monaco_susceptibility_v2.tif",
            "uncertainty": root
            / "outputs/demonstration_v2/waimea_monaco_uncertainty_v2.tif",
            "threshold": root
            / "outputs/validation_v2/selected_threshold_v2.json",
            "checkpoint": root / "models/climateadapt_unet_v2_best.pt",
            "land_cover": root
            / "data/processed/combined_rasters/land_cover_class_5m.tif",
        }

        missing = [str(path.relative_to(root)) for path in paths.values() if not path.exists()]
        if missing:
            state.fail(self.name, "Missing required files: " + ", ".join(missing))
            return state

        with paths["summary"].open("r", encoding="utf-8") as file:
            summary = json.load(file)
        with paths["threshold"].open("r", encoding="utf-8") as file:
            threshold_record = json.load(file)

        safety_checks = {
            "flood_target_raster_accessed": summary.get(
                "flood_target_raster_accessed"
            ),
            "model_retrained": summary.get("model_retrained"),
            "threshold_modified": summary.get("threshold_modified"),
        }
        if any(value is not False for value in safety_checks.values()):
            state.fail(
                self.name,
                "Demonstration provenance checks failed: " + str(safety_checks),
            )
            return state

        summary_threshold = float(summary["operational_threshold"])
        frozen_value = threshold_record.get(
            "operational_threshold",
            threshold_record.get("selected_threshold"),
        )
        if frozen_value is None:
            state.fail(self.name, "No frozen threshold was found in validation JSON.")
            return state
        frozen_threshold = float(frozen_value)
        if not np.isclose(summary_threshold, frozen_threshold, atol=1e-8):
            state.fail(
                self.name,
                f"Threshold mismatch: demonstration={summary_threshold}, "
                f"validation={frozen_threshold}.",
            )
            return state

        raster_metadata = {}
        reference_signature = None
        for key in ["probability", "binary", "uncertainty", "land_cover"]:
            with rasterio.open(paths[key]) as source:
                signature = (
                    source.width,
                    source.height,
                    tuple(source.transform),
                    source.crs.to_string() if source.crs else None,
                )
                raster_metadata[key] = {
                    "width": source.width,
                    "height": source.height,
                    "crs": source.crs.to_string() if source.crs else None,
                    "resolution": [abs(source.transform.a), abs(source.transform.e)],
                    "nodata": source.nodata,
                }
                if reference_signature is None:
                    reference_signature = signature
                elif signature != reference_signature:
                    state.fail(self.name, f"Raster alignment mismatch for {key}.")
                    return state

        if raster_metadata["probability"]["crs"] != "EPSG:2193":
            state.fail(
                self.name,
                "Expected NZTM2000 / EPSG:2193 demonstration rasters.",
            )
            return state

        if int(summary["uncovered_cells"]) != 0:
            state.warnings.append(
                f"The demonstration contains {summary['uncovered_cells']} uncovered cells."
            )

        state.facts["validated_summary"] = summary
        state.facts["data_validation"] = {
            "status": "passed",
            "frozen_operational_threshold": frozen_threshold,
            "provenance_checks": safety_checks,
            "raster_metadata": raster_metadata,
            "all_rasters_aligned": True,
            "crs_valid": True,
        }
        state.artifacts.update(
            {key: str(path.relative_to(root)) for key, path in paths.items()}
        )
        state.status = "data_validated"
        state.log(self.name, "validate_inputs", "passed")
        return state
