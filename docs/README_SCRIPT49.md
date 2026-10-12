# Script 49 – V3 area-based Place analysis

This update restores the useful V2-style place workflow while retaining the
stronger V3 logistic model and its frozen `0.004` threshold.

## What changes

- Place analysis evaluates a **500 m, 1 km or 2 km radius**, not one raster cell.
- The page reports evaluated land, flagged land area and flagged percentage.
- A local four-panel map shows elevation, susceptibility score, flagged land and
  distance to the coastline.
- The V3 score is consistently described as an **uncalibrated screening score**,
  not a flood probability.
- The result flows through the Data Validation, Flood Analysis and Planning
  Support agents before the mandatory human-review checkpoint.
- Rocks Road and Monaco have fixed offline catalogue entries so those demo
  searches are deterministic.

## Files to copy into the project

Copy these files from the extracted bundle, preserving their folders:

```text
dashboard/app.py
src/climateadapt/place_analysis.py
src/climateadapt/agent/live_place_workflow_v1.py
scripts/49_test_v3_area_analysis.py
```

## Run the test

Keep the V3 prediction API running in one PowerShell window:

```powershell
python .\scripts\30_run_prediction_api.py
```

Then run:

```powershell
python .\scripts\49_test_v3_area_analysis.py
```

## Run the dashboard

Keep Ollama available for the Planning Support Agent, then start Streamlit:

```powershell
streamlit run .\dashboard\app.py
```

Open **Place analysis**, search for **Rocks Road**, select **1 kilometre**, and
choose **Analyse area with the three agents**.

## Interpretation

The flagged percentage is the share of model-covered land cells in the selected
radius whose V3 score meets the frozen threshold. It is not the probability that
the place will flood. The output supports prioritisation and human review only.
