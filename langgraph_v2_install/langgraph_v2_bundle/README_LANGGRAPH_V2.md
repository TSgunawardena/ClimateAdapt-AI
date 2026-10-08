# ClimateAdapt AI – LangGraph Workflow V2

Copy the `src` and `scripts` folders into the project root. Existing V1 files
are not replaced.

Install the orchestration dependency:

```powershell
python -m pip install "langgraph>=1.2,<2"
```

Add the following line to `requirements.txt`:

```text
langgraph>=1.2,<2
```

Run interactively:

```powershell
python .\scripts\24_run_langgraph_workflow_v2.py --reviewer "Your name"
```

At the checkpoint, choose `approve`, `revise`, or `reject`. Choosing `revise`
routes back to the Planning Support Agent and then pauses for review again.

For a repeatable classroom demonstration:

```powershell
python .\scripts\24_run_langgraph_workflow_v2.py --auto-approve --reviewer "Prototype demo"
```

The workflow reads already-created U-Net, baseline, MC-dropout and Physicalist
artifacts. It does not retrain models or change the frozen threshold.

