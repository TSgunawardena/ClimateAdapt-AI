# ClimateAdapt AI V3 — Agent Compliance Record

## What counts as the AI agent

The prototype contains a three-agent LangGraph workflow. One of the three is a
genuine LLM-powered AI agent; the other two are deterministic specialist agents.

| Agent | Implementation | Goal | Stopping condition |
|---|---|---|---|
| Data Validation Agent | Deterministic Python/LangGraph node | Accept only aligned, covered, provenance-rich V3 results | Stop the workflow immediately when validation fails |
| Flood Analysis Agent | Deterministic Python/LangGraph node | Convert verified scores and terrain fields into constrained evidence | Pass structured evidence forward without making an external decision |
| Planning Support Agent | **Fine-tuned Qwen3 4B LLM through Ollama** | Draft a schema-constrained planning brief from supplied evidence | Stop at mandatory human review; use a deterministic fallback when the LLM fails its safety gate |

The Planning Support Agent is not merely a hard-coded function. It sends a
system prompt and structured task to the fine-tuned
`climateadapt-planner:4b` model through Ollama `/api/chat`, receives generated
JSON, and validates that JSON against implementable safety rules.

## Tools and actions

- Prediction tool: FastAPI `/predict/point` using the frozen V3 logistic
  susceptibility raster.
- Model backend selector: Ollama, with both `qwen3:4b` and
  `climateadapt-planner:4b` registered by model name.
- Orchestration and state: LangGraph with checkpointed state and conditional
  routes.
- Real external artefact: a private Gitea issue created only after recorded
  approval and independently read back for verification.

## Implementable human-approval rule

```text
IF human_decision.action != "approve":
    Gitea publication is prohibited.
```

The routing layer applies the rule and the publication node checks it again.
Reject ends without an external action. Revise returns to the Planning Support
Agent. Invalid actions default to rejection.

## What the agent must never do autonomously

- Approve or refuse land use.
- Issue an emergency or evacuation instruction.
- Make an engineering, funding or regulatory decision.
- Present the V3 susceptibility score as flood probability.
- Release a planning brief publicly without authorised human review.

## Platform substitution

The implementation uses tutor-approved alternatives to the supplied
EduStack/OpenClaw path: LangGraph for orchestration, Ollama for local model
serving and Gitea for the real artefact. The report must describe this as a
substitution and must not claim that OpenClaw or EduStack was used.

## Evaluation evidence

Script 52 records, for every normal run, the model name, generated-token count,
accepted LLM safety-gate result and complete three-agent audit trace. It also
tests an unavailable LLM, requiring deterministic fallback and the same human
checkpoint. This distinguishes the LLM-powered Planning Support Agent from the
two deterministic agents.
