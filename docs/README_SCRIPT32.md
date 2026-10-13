# ClimateAdapt AI — Script 32 Verified Gitea Workflow

This extends Script 31. After recorded human approval, the LangGraph workflow
creates a Gitea issue. A separate node performs an independent GET request and
verifies the persisted issue number, title, body hash, run ID, approval text,
repository and state.

## Required existing files

Script 31 must already be installed:

```text
src/climateadapt/agent/live_place_workflow_v1.py
```

## Copy files

Copy these files into the matching project directories:

```text
scripts/32_run_gitea_agent_workflow.py
src/climateadapt/agent/gitea_workflow_v1.py
```

## Required running services

1. Gitea on `http://127.0.0.1:3000`
2. Script 30 prediction API on `http://127.0.0.1:8000`
3. Ollama with `climateadapt-planner:4b`

## Environment variables

Set the URL, owner and repository in the workflow PowerShell window:

```powershell
$env:CLIMATEADAPT_GITEA_URL = "http://127.0.0.1:3000"
$env:CLIMATEADAPT_GITEA_OWNER = "tharusha"
$env:CLIMATEADAPT_GITEA_REPO = "climateadapt-ai"
```

Enter the token without displaying it:

```powershell
$secureToken = Read-Host "Paste the Gitea token" -AsSecureString
$env:CLIMATEADAPT_GITEA_TOKEN = `
  [System.Net.NetworkCredential]::new("", $secureToken).Password
```

## Run

```powershell
python .\scripts\32_run_gitea_agent_workflow.py `
  --latlon -41.29 173.24 `
  --name "Nelson example" `
  --reviewer "Tharusha"
```

At the checkpoint, `approve` creates a real issue. `reject` safely stops with no
Gitea write. `revise` returns the brief to the planning agent.

## Evidence

The workflow stores no token. It writes:

```text
outputs/gitea_agent_workflow_v1/gitea_evidence_<run-id>.json
outputs/gitea_agent_workflow_v1/workflow_<run-id>.json
outputs/gitea_agent_workflow_v1/planning_brief_<run-id>.md
```
