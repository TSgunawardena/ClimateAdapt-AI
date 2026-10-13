# ClimateAdapt AI — Script 52 V3 Composed-System Evaluation

Script 52 evaluates the completed V3 system over nine cases defined before the
run:

- three Nelson locations repeated twice;
- an out-of-region request;
- an unavailable prediction API; and
- an unavailable planning LLM.

For every normal run it requires evidence that:

- the API returned the V3 logistic contract and frozen threshold `0.004`;
- the Data Validation, Flood Analysis and Planning Support Agents all executed;
- the fine-tuned Ollama model generated tokens and its schema-constrained
  response passed the safety gate;
- LangGraph reached the human checkpoint;
- the predeclared human rejection prevented a Gitea write; and
- repeated requests returned identical susceptibility scores and consistent
  planning statuses.

The LLM-outage case must use the deterministic safe fallback and still stop at
human review. Gitea issue identifiers and counts are independently compared
before and after the run.

## Run

```powershell
python .\scripts\52_evaluate_v3_composed_system.py
```

This is deliberately non-destructive. Every interrupted run is rejected by the
test harness, so no Gitea issue may be created.

## Outputs

```text
outputs/composed_system_evaluation_v3/v3_composed_system_test_results.csv
outputs/composed_system_evaluation_v3/v3_composed_system_evaluation_summary.json
evidence/figures/v3/v3_composed_system_evaluation.png
```

The evaluation measures orchestration, safety and service behaviour. It does
not estimate flood-prediction accuracy; that evidence comes from the locked V3
geographic test.
