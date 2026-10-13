# ClimateAdapt AI — Script 51 V3 Robustness Audit

Script 51 reruns the non-destructive robustness and safe-failure audit against
the final V3 service contract.

It verifies:

- `/health` exposes the EPSG:2193 CPU raster service and V3 rasters;
- `/model` identifies the V3 logistic model, threshold `0.004`, uncalibrated
  score semantics and label-blind serving;
- a valid point request returns coverage, provenance and mandatory human review;
- incomplete, conflicting, out-of-range, malformed and out-of-region requests
  fail safely;
- prediction and Ollama service outages are detected;
- the fine-tuned `climateadapt-planner:4b` model is registered in Ollama;
- LangGraph routes approve, revise and reject decisions correctly;
- the Gitea publication node blocks any state without recorded approval;
- Gitea is checked read-only, missing credentials fail before network access,
  and saved workflow evidence does not contain the token.

The audit creates no Gitea issue and changes no model, feature, split or
threshold.

## Prerequisites

Run these services first:

1. Gitea at `http://127.0.0.1:3000`.
2. The V3 prediction API using Script 30.
3. Ollama with `climateadapt-planner:4b`.

Set the four `CLIMATEADAPT_GITEA_*` environment variables in the same
PowerShell session. The token is read only from the environment.

## Run

```powershell
python .\scripts\51_run_v3_robustness_audit.py
```

## Outputs

```text
outputs/robustness_audit_v3/v3_robustness_test_results.csv
outputs/robustness_audit_v3/v3_robustness_audit_summary.json
evidence/figures/v3/v3_robustness_audit.png
```
