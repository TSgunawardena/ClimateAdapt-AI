# ClimateAdapt AI Streamlit Dashboard

This dashboard presents the frozen Experiment 2 evidence and the outputs of the
three-agent workflow. It does not retrain the model or change the threshold.

## Install

Copy the contents of this bundle into the root of `ClimateAdapt_AI_Starter` and
merge the folders. Streamlit must be installed in the existing virtual
environment.

## Check

```powershell
python .\scripts\15_check_dashboard.py
```

## Run

```powershell
streamlit run .\dashboard\app.py
```

The browser should open at `http://localhost:8501`.

## Dashboard sections

- Overview and final demonstration map
- Geospatial output inventory
- Three-agent status and audit trail
- Training, validation and independent test evidence
- Controlled human-review checkpoint
- Downloadable reports

Human approval in this prototype means only that an output may proceed to
further investigation. It never authorises emergency, regulatory, funding,
construction, land-use or public-warning action.
