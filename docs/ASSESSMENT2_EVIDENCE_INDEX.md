# Assessment 2 Evidence Index

This index identifies the strongest prototype evidence and the claim supported
by each item. Preserve terminal screenshots alongside the generated files.

## Data preparation and geographic separation

| Evidence | Supported claim |
| --- | --- |
| `evidence/figures/full_region_cleaned_vectors.png` | Invalid coastal-label geometry was repaired before rasterisation. |
| `evidence/figures/full_region_rasters_overview.png` | Full Nelson input layers were aligned to a common 5 m grid. |
| `evidence/figures/combined_rasters_and_split.png` | Training, validation, test and demonstration regions are geographically separated. |
| `evidence/figures/tile_distribution.png` | Tile and class distributions were recorded. |

## Model training and independent testing

| Evidence | Supported claim |
| --- | --- |
| `evidence/figures/unet_v2_training_curves.png` | Training history and validation checkpoint selection. |
| `evidence/figures/validation_threshold_selection_v2.png` | Operational threshold selected only on Rocks Road validation data. |
| `outputs/test_v2/tahunanui_test_metrics_v2.json` | Frozen-model independent Tāhunanui results. |
| `evidence/figures/tahunanui_test_evaluation_v2.png` | Spatial true/false prediction pattern. |
| `evidence/figures/tahunanui_test_evaluation_curves_v2.png` | ROC and precision–recall evaluation. |

## Baseline, uncertainty and physical reasoning

| Evidence | Supported claim |
| --- | --- |
| `evidence/figures/baseline_vs_unet_v2.png` | Honest comparison with trivial and logistic baselines. |
| `evidence/figures/tahunanui_mc_dropout_uncertainty_v2.png` | Epistemic uncertainty is associated with elevated error. |
| `evidence/figures/tahunanui_physicalist_guardrail_v2.png` | Terrain/connectivity checks support explainable human review. |

## Regional deployment and user demonstration

| Evidence | Supported claim |
| --- | --- |
| `evidence/figures/nelson_full_region_inference_v2.png` | Label-blind full-region susceptibility screening. |
| `evidence/figures/waimea_monaco_demonstration_v2.png` | Unseen-location planning demonstration without model retuning. |
| Prediction API `/docs` screenshot | Instant location query interface. |
| Streamlit dashboard screenshots | Accessible prototype user interface. |

## Agentic workflow and language-model evidence

| Evidence | Supported claim |
| --- | --- |
| `outputs/live_agent_workflow_v1/` | Three-agent LangGraph workflow and human checkpoint. |
| `outputs/gitea_agent_workflow_v1/` | Human-approved Gitea action plus independent verification. |
| Gitea issue screenshot | Persisted external action visible in the private repository. |
| `evidence/figures/base_vs_finetuned_language_test.png` | Held-out base-versus-fine-tuned LLM comparison. |
| `outputs/language_model_comparison_v1/base_vs_finetuned_test_summary.json` | Exact language-model test metrics. |

## Explainability, bias and safe failure

| Evidence | Supported claim |
| --- | --- |
| `evidence/figures/explainability_bias_audit_v2.png` | Grouped feature ablation and environmental subgroup disparities. |
| `outputs/explainability_bias_v2/explainability_bias_audit_v2.json` | Bias warnings, unseen classes and required controls. |
| `evidence/figures/robustness_audit.png` | Summary of 17 safe-failure checks. |
| `outputs/robustness_audit_v1/robustness_test_results.csv` | Exact expected and observed result for every robustness test. |
| `outputs/robustness_audit_v1/robustness_audit_summary.json` | 17 passed, zero failed, zero Gitea writes and zero leaked secrets. |

## Mandatory interpretation statement

ClimateAdapt AI is a screening-level research prototype. It is not a
hydrodynamic simulation, emergency-warning product, engineering design tool or
regulatory decision system. Model outputs require GIS, engineering, community
and authorised human review.

