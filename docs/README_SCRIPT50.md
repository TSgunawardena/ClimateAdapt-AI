# ClimateAdapt AI — Script 50 V3 Explainability and Data-Bias Audit

Script 50 audits the **deployed V3 logistic-regression screening model**. It
does not train a model or change the frozen operational threshold of `0.004`.

## What it adds

- Exact logistic-model attribution using `encoded feature × fitted coefficient`.
- Mean absolute logit contribution on the already-unlocked V3 test pixels.
- Geographic checks for Delaware Open Coast and Oananga Bay.
- Environmental subgroup checks for elevation, slope, coast distance and the
  eight engineered land-cover groups.
- Predeclared review triggers for large recall or precision ranges.
- CSV, JSON and figure evidence suitable for the assessment report.

This is a transparent-model equivalent to post-hoc attribution: it shows how
the fitted logistic model uses its encoded inputs. It **does not prove
causation**, physical flood mechanisms, future generalisation or an absence of
bias. The subgroup audit concerns environmental/spatial data bias; it is not a
demographic fairness audit.

## Required existing project artifacts

The script expects the completed V3 pipeline, including:

```text
models/v3/logistic_regression_v3.joblib
data/processed/v3/v3_split_mask_5m.tif
data/processed/v3/tiles/v3_tile_manifest.csv
data/processed/v3/tiles/v3_tile_metadata.json
data/processed/v3/tiles/test/*.npz
outputs/baselines_v3/logistic_selected_threshold_v3.json
outputs/test_v3/v3_locked_test_summary.json
```

## Install

Extract the ZIP into a temporary folder, then copy its `scripts` and `docs`
folders into the project root. From PowerShell in the project root:

```powershell
$zip = ".\ClimateAdapt_AI_Script50_V3_Explainability_Bias.zip"
$stage = ".\script50_stage"

Expand-Archive -Path $zip -DestinationPath $stage -Force
Copy-Item "$stage\scripts\*" ".\scripts\" -Force
Copy-Item "$stage\docs\*" ".\docs\" -Force
```

## Run

```powershell
python .\scripts\50_audit_v3_logistic_explainability_bias.py
```

## Outputs

```text
outputs/explainability_bias_v3/v3_logistic_exact_attribution.csv
outputs/explainability_bias_v3/v3_environmental_subgroup_metrics.csv
outputs/explainability_bias_v3/v3_subgroup_performance_ranges.csv
outputs/explainability_bias_v3/v3_explainability_bias_audit.json
evidence/figures/v3/v3_logistic_explainability_bias_audit.png
```

The test regions were already opened for final evaluation by Script 44. Script
50 reuses them strictly for post-hoc auditing. Do not use its results to alter
features, model weights, split membership or thresholds.
