# ClimateAdapt AI – Three-Agent Prototype Setup

This bundle adds three deterministic, auditable agents to the existing
ClimateAdapt AI project. It uses the frozen Experiment 2 Waimea/Monaco outputs
and does not require a paid API or language model.

## Agents

1. **Data Validation Agent** – checks files, CRS, alignment, provenance and the
   frozen validation threshold.
2. **Flood Analysis Agent** – derives susceptibility, probability, uncertainty
   and land-cover statistics from the model outputs.
3. **Planning Support Agent** – creates guarded planning observations and stops
   at a mandatory human-review checkpoint.

## Install

Copy the `src`, `scripts` and `README_AGENT_SETUP.md` items from this bundle into
the root of `ClimateAdapt_AI_Starter`. Merge the folders when Windows asks; do
not delete the existing project files.

The workflow uses packages already installed for the ML pipeline: NumPy,
Rasterio and PyTorch.

## Run

Activate the existing environment and run:

```powershell
python .\scripts\14_run_agent_workflow.py
```

## Expected outputs

```text
outputs\agent_workflow_v1\workflow_state.json
outputs\agent_workflow_v1\waimea_monaco_agent_report.json
outputs\agent_workflow_v1\waimea_monaco_agent_report.md
```

The final status should be `human_review_required`. This is intentional: the
prototype must not make autonomous emergency, regulatory, funding, land-use or
cultural-value decisions.
