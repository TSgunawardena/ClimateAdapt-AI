"""Final paired comparison of base and fine-tuned ClimateAdapt language models.

This script opens the held-out test split for the first and final comparison.
Both models receive identical prompts and deterministic generation settings.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "instruction" / "climateadapt_instruction_test.jsonl"
BASELINE_SCRIPT = ROOT / "scripts" / "27_evaluate_base_llm.py"
OUTPUT_DIR = ROOT / "outputs" / "language_model_comparison_v1"
FIGURE_DIR = ROOT / "evidence" / "figures"

MODELS = {
    "base": "qwen3:4b",
    "fine_tuned": "climateadapt-planner:4b",
}


def load_evaluator():
    if not BASELINE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Required evaluator is missing: {BASELINE_SCRIPT.relative_to(ROOT)}"
        )
    specification = importlib.util.spec_from_file_location(
        "climateadapt_language_evaluator", BASELINE_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load Script 27 evaluation functions.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing held-out test file: {path.relative_to(ROOT)}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
    return records


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            result = json.loads(line)
            completed[result["id"]] = result
    return completed


def unload_model(evaluator, model_name: str) -> None:
    try:
        evaluator.post_json(
            "/api/generate",
            {"model": model_name, "keep_alive": 0},
            timeout=30,
        )
    except Exception as error:  # Unloading is helpful but not evaluation-critical.
        print(f"Warning: could not explicitly unload {model_name}: {error}")


def evaluate_model(
    evaluator,
    model_key: str,
    model_name: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluator.MODEL = model_name
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    response_path = OUTPUT_DIR / f"{model_slug}_test_responses.jsonl"
    completed = load_completed(response_path)

    show = evaluator.check_ollama()
    details = show.get("details", {})
    print("-" * 78)
    print(f"Model role:       {model_key}")
    print(f"Model:            {model_name}")
    print(f"Family:           {details.get('family', 'unknown')}")
    print(f"Parameters:       {details.get('parameter_size', 'unknown')}")
    print(f"Quantization:     {details.get('quantization_level', 'unknown')}")
    print(f"Completed/resume: {len(completed)}/{len(records)}")

    mode = "a" if response_path.exists() else "w"
    with response_path.open(mode, encoding="utf-8", newline="\n") as output:
        for index, record in enumerate(records, start=1):
            if record["id"] in completed:
                print(f"[{index:02d}/{len(records)}] {record['id']} – already completed")
                continue

            print(f"[{index:02d}/{len(records)}] {record['id']} – requesting {model_name}...")
            response, elapsed = evaluator.request_model(record["messages"][:2])
            message = response.get("message", {})
            content = str(message.get("content", ""))
            thinking = str(message.get("thinking", ""))
            metrics = evaluator.score_response(record, content, thinking)

            result = {
                "id": record["id"],
                "split": "held_out_test",
                "source_type": record["source_type"],
                "task_type": record["task_type"],
                "expected_status": record["metadata"]["expected_status"],
                "model_role": model_key,
                "model": response.get("model", model_name),
                "raw_content": content,
                "parsed_content": evaluator.parse_json_object(content),
                "thinking_content": thinking,
                "metrics": metrics,
                "runtime": {
                    "elapsed_seconds": elapsed,
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "eval_count": response.get("eval_count"),
                    "done_reason": response.get("done_reason"),
                },
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            completed[record["id"]] = result
            print(
                f"  status={metrics['status_accuracy']:.0f} "
                f"guardrail={metrics['task_guardrail_accuracy']:.0f} "
                f"numbers={metrics['numeric_fidelity']:.2f} "
                f"behaviour={metrics['behaviour_score']:.2f} "
                f"time={elapsed:.1f}s"
            )

    ordered = [completed[record["id"]] for record in records]
    aggregate = evaluator.aggregate(ordered)
    total_elapsed = sum(float(item["runtime"]["elapsed_seconds"]) for item in ordered)
    total_tokens = sum(int(item["runtime"].get("eval_count") or 0) for item in ordered)

    model_summary = {
        "model": model_name,
        "model_role": model_key,
        "details": details,
        "examples": len(ordered),
        "metrics": aggregate,
        "runtime": {
            "total_elapsed_seconds": total_elapsed,
            "mean_elapsed_seconds": total_elapsed / len(ordered),
            "total_output_tokens": total_tokens,
            "output_tokens_per_second": total_tokens / total_elapsed if total_elapsed else 0.0,
        },
        "response_file": str(response_path.relative_to(ROOT)),
    }
    unload_model(evaluator, model_name)
    return ordered, model_summary


def task_breakdown(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["task_type"]].append(result)
    return {
        task: {
            "examples": len(items),
            "status_accuracy": mean(item["metrics"]["status_accuracy"] for item in items),
            "guardrail_accuracy": mean(
                item["metrics"]["task_guardrail_accuracy"] for item in items
            ),
            "behaviour_score": mean(item["metrics"]["behaviour_score"] for item in items),
        }
        for task, items in sorted(grouped.items())
    }


def create_figure(
    base_metrics: dict[str, float], fine_tuned_metrics: dict[str, float]
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    selected = [
        ("JSON parse", "parse_success"),
        ("Schema", "schema_completeness"),
        ("Status", "status_accuracy"),
        ("Approval", "human_approval_accuracy"),
        ("Guardrail", "task_guardrail_accuracy"),
        ("Numbers", "numeric_fidelity"),
        ("Behaviour", "behaviour_score"),
    ]
    labels = [label for label, _ in selected]
    base_values = [base_metrics[key] for _, key in selected]
    tuned_values = [fine_tuned_metrics[key] for _, key in selected]
    x = np.arange(len(labels))
    width = 0.36

    figure, axis = plt.subplots(figsize=(13, 6.5))
    base_bars = axis.bar(x - width / 2, base_values, width, label="Base Qwen3 4B", color="#7a8ca5")
    tuned_bars = axis.bar(x + width / 2, tuned_values, width, label="ClimateAdapt fine-tuned", color="#2a9d8f")
    axis.bar_label(base_bars, labels=[f"{value:.2f}" for value in base_values], padding=3, fontsize=8)
    axis.bar_label(tuned_bars, labels=[f"{value:.2f}" for value in tuned_values], padding=3, fontsize=8)
    axis.set_ylim(0, 1.10)
    axis.set_ylabel("Held-out test score")
    axis.set_title("ClimateAdapt Language Model — Base vs Fine-tuned", weight="bold")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "base_vs_finetuned_language_test.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator()
    records = read_jsonl(TEST_FILE)
    if len(records) != 27:
        raise ValueError(f"Expected 27 held-out test examples, found {len(records)}")

    print("=" * 78)
    print("CLIMATEADAPT – FINAL BASE VS FINE-TUNED LANGUAGE MODEL COMPARISON")
    print("=" * 78)
    print(f"Held-out test examples: {len(records)}")
    print("Models:                 qwen3:4b vs climateadapt-planner:4b")
    print("Thinking:               disabled for both")
    print("Temperature:            0 for both")
    print("Context/output tokens:  4096 / 700 for both")
    print("Seed:                    8532 for both")
    print("Fine-tuning changes:     frozen model weights plus trained LoRA merge")
    print("No further tuning will be performed from test results.")

    results_by_role: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for role, model_name in MODELS.items():
        results, summary = evaluate_model(evaluator, role, model_name, records)
        results_by_role[role] = results
        summaries[role] = summary

    paired_path = OUTPUT_DIR / "paired_test_results.csv"
    metric_names = list(summaries["base"]["metrics"])
    with paired_path.open("w", encoding="utf-8", newline="") as output:
        fields = [
            "id",
            "source_type",
            "task_type",
            "expected_status",
            "base_predicted_status",
            "fine_tuned_predicted_status",
            *[f"base_{metric}" for metric in metric_names],
            *[f"fine_tuned_{metric}" for metric in metric_names],
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            base = results_by_role["base"][index]
            tuned = results_by_role["fine_tuned"][index]
            writer.writerow(
                {
                    "id": record["id"],
                    "source_type": record["source_type"],
                    "task_type": record["task_type"],
                    "expected_status": record["metadata"]["expected_status"],
                    "base_predicted_status": (base.get("parsed_content") or {}).get("status", "PARSE_FAILED"),
                    "fine_tuned_predicted_status": (tuned.get("parsed_content") or {}).get("status", "PARSE_FAILED"),
                    **{f"base_{metric}": base["metrics"][metric] for metric in metric_names},
                    **{f"fine_tuned_{metric}": tuned["metrics"][metric] for metric in metric_names},
                }
            )

    base_metrics = summaries["base"]["metrics"]
    tuned_metrics = summaries["fine_tuned"]["metrics"]
    improvements = {
        metric: tuned_metrics[metric] - base_metrics[metric]
        for metric in metric_names
    }

    real_ids = {
        record["id"]
        for record in records
        if record["source_type"] == "real_held_out"
    }
    real_case_results = []
    for index, record in enumerate(records):
        if record["id"] not in real_ids:
            continue
        base = results_by_role["base"][index]
        tuned = results_by_role["fine_tuned"][index]
        real_case_results.append(
            {
                "id": record["id"],
                "task_type": record["task_type"],
                "expected_status": record["metadata"]["expected_status"],
                "base_status": (base.get("parsed_content") or {}).get("status", "PARSE_FAILED"),
                "fine_tuned_status": (tuned.get("parsed_content") or {}).get("status", "PARSE_FAILED"),
                "base_behaviour_score": base["metrics"]["behaviour_score"],
                "fine_tuned_behaviour_score": tuned["metrics"]["behaviour_score"],
            }
        )

    summary = {
        "experiment": "ClimateAdapt final paired language-model comparison",
        "evaluation_split": "held-out test",
        "held_out_test_examples": len(records),
        "real_project_examples": len(real_ids),
        "generation_settings_identical": True,
        "generation_configuration": {
            "thinking": False,
            "temperature": 0,
            "seed": evaluator.SEED,
            "context_length": evaluator.CONTEXT_LENGTH,
            "maximum_output_tokens": evaluator.MAX_OUTPUT_TOKENS,
            "format": "json",
        },
        "models": summaries,
        "absolute_improvement_fine_tuned_minus_base": improvements,
        "task_breakdown": {
            role: task_breakdown(results)
            for role, results in results_by_role.items()
        },
        "real_held_out_case_results": real_case_results,
        "success_criteria": {
            "fine_tuned_status_accuracy_at_least_0_80": tuned_metrics["status_accuracy"] >= 0.80,
            "fine_tuned_guardrail_accuracy_at_least_0_80": tuned_metrics["task_guardrail_accuracy"] >= 0.80,
            "fine_tuned_parse_success_at_least_0_95": tuned_metrics["parse_success"] >= 0.95,
            "fine_tuned_human_approval_at_least_0_95": tuned_metrics["human_approval_accuracy"] >= 0.95,
            "fine_tuned_numeric_fidelity_at_least_0_95": tuned_metrics["numeric_fidelity"] >= 0.95,
        },
        "important_note": (
            "The held-out test split was opened only for this final paired comparison. "
            "No model, prompt, threshold or training setting will be changed using these results."
        ),
    }
    summary_path = OUTPUT_DIR / "base_vs_finetuned_test_summary.json"
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)

    figure_path = create_figure(base_metrics, tuned_metrics)

    print("\n" + "=" * 78)
    print("FINAL HELD-OUT LANGUAGE MODEL COMPARISON COMPLETED")
    print("=" * 78)
    print(f"{'Metric':<30} {'Base':>10} {'Fine-tuned':>12} {'Change':>10}")
    print("-" * 78)
    for metric in metric_names:
        print(
            f"{metric.replace('_', ' ').title():<30} "
            f"{base_metrics[metric]:>10.4f} "
            f"{tuned_metrics[metric]:>12.4f} "
            f"{improvements[metric]:>+10.4f}"
        )
    print("-" * 78)
    print("Paired results:", paired_path.relative_to(ROOT))
    print("Summary:       ", summary_path.relative_to(ROOT))
    if figure_path is not None:
        print("Figure:        ", figure_path.relative_to(ROOT))
    print("=" * 78)
    print("No further model or prompt tuning should be performed using test results.")


if __name__ == "__main__":
    main()
