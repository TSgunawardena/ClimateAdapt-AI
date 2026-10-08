"""Agent 2: derive factual susceptibility statistics from model rasters."""

from __future__ import annotations

from collections import Counter

import numpy as np
import rasterio

from .state import WorkflowState


class FloodAnalysisAgent:
    name = "flood_analysis_agent"

    def run(self, state: WorkflowState) -> WorkflowState:
        state.current_agent = self.name
        if state.status != "data_validated":
            state.fail(self.name, "Data validation must pass before flood analysis.")
            return state

        state.status = "analysing_susceptibility"
        root = state.root
        summary = state.facts["validated_summary"]
        threshold = float(summary["operational_threshold"])

        probability_path = root / state.artifacts["probability"]
        binary_path = root / state.artifacts["binary"]
        uncertainty_path = root / state.artifacts["uncertainty"]
        land_cover_path = root / state.artifacts["land_cover"]

        with rasterio.open(probability_path) as source:
            probability = source.read(1).astype(np.float32)
            probability_valid = probability != source.nodata
            transform = source.transform
        with rasterio.open(binary_path) as source:
            binary = source.read(1)
            binary_valid = binary != source.nodata
        with rasterio.open(uncertainty_path) as source:
            uncertainty = source.read(1).astype(np.float32)
            uncertainty_valid = uncertainty != source.nodata
        with rasterio.open(land_cover_path) as source:
            land_cover = source.read(1)

        valid = probability_valid & binary_valid & uncertainty_valid
        if not np.any(valid):
            state.fail(self.name, "No common valid demonstration pixels were found.")
            return state

        probabilities = probability[valid]
        uncertainties = uncertainty[valid]
        susceptible = valid & (binary == 1)
        susceptible_cells = int(np.sum(susceptible))
        valid_cells = int(np.sum(valid))
        cell_area_m2 = abs(float(transform.a) * float(transform.e))

        risk_bands = {
            "lower_indicator_below_0_35": int(np.sum(probabilities < 0.35)),
            "review_band_0_35_to_threshold": int(
                np.sum((probabilities >= 0.35) & (probabilities < threshold))
            ),
            "operational_susceptibility": int(np.sum(probabilities >= threshold)),
            "high_indicator_0_75_or_above": int(np.sum(probabilities >= 0.75)),
        }
        risk_band_percentages = {
            key: 100.0 * value / valid_cells for key, value in risk_bands.items()
        }

        class_counts = Counter(int(value) for value in land_cover[susceptible])
        leading_land_cover = [
            {
                "class_code": class_code,
                "cells": count,
                "percentage_of_susceptible_cells": 100.0 * count / susceptible_cells,
            }
            for class_code, count in class_counts.most_common(5)
        ] if susceptible_cells else []

        quantile_values = np.quantile(
            probabilities, [0.05, 0.25, 0.50, 0.75, 0.95]
        )
        analysis = {
            "analysis_type": "label-blind susceptibility analysis",
            "accuracy_claimed": False,
            "operational_threshold": threshold,
            "evaluated_cells": valid_cells,
            "evaluated_area_square_kilometres": (
                valid_cells * cell_area_m2 / 1_000_000.0
            ),
            "susceptible_cells": susceptible_cells,
            "susceptible_area_square_kilometres": (
                susceptible_cells * cell_area_m2 / 1_000_000.0
            ),
            "susceptible_percentage": 100.0 * susceptible_cells / valid_cells,
            "probability_quantiles": {
                "p05": float(quantile_values[0]),
                "p25": float(quantile_values[1]),
                "p50": float(quantile_values[2]),
                "p75": float(quantile_values[3]),
                "p95": float(quantile_values[4]),
            },
            "risk_band_cells": risk_bands,
            "risk_band_percentages": risk_band_percentages,
            "mean_uncertainty_indicator": float(np.mean(uncertainties)),
            "high_uncertainty_cells_entropy_0_80_or_above": int(
                np.sum(uncertainties >= 0.80)
            ),
            "high_uncertainty_percentage": float(
                100.0 * np.sum(uncertainties >= 0.80) / valid_cells
            ),
            "leading_susceptible_land_cover_codes": leading_land_cover,
        }

        recorded_cells = int(summary["susceptible_cells"])
        if susceptible_cells != recorded_cells:
            state.warnings.append(
                f"Calculated susceptible cells ({susceptible_cells}) differ from "
                f"the demonstration summary ({recorded_cells})."
            )

        state.facts["flood_analysis"] = analysis
        state.status = "susceptibility_analysed"
        state.log(self.name, "analyse_model_outputs", "completed")
        return state
