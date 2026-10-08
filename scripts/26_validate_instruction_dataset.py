"""Audit the ClimateAdapt instruction dataset before LLM fine-tuning."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "instruction"
OUTPUT_DIR = ROOT / "outputs" / "instruction_dataset"
FIGURE_DIR = ROOT / "evidence" / "figures"

FILES = {
    "train": DATASET_DIR / "climateadapt_instruction_train.jsonl",
    "validation": DATASET_DIR / "climateadapt_instruction_validation.jsonl",
    "test": DATASET_DIR / "climateadapt_instruction_test.jsonl",
}

EXPECTED_ROLES = ["system", "user", "assistant"]
EXPECTED_STATUSES = {
    "blocked_data_validation",
    "draft_ready_for_human_review",
    "human_review_required",
}
REQUIRED_TARGET_FIELDS = {
    "status",
    "headline",
    "summary",
    "evidence",
    "recommended_actions",
    "limitations",
    "requires_human_approval",
    "prohibited_automatic_actions",
    "stopping_condition",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path.relative_to(ROOT)}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}: {error}"
                ) from error
            records.append(record)
    return records


def fingerprint(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_record(record: dict[str, Any], expected_split: str) -> list[str]:
    errors: list[str] = []
    record_id = str(record.get("id", "<missing-id>"))

    if record.get("split") != expected_split:
        errors.append(
            f"{record_id}: split field is {record.get('split')!r}, expected {expected_split!r}"
        )

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"{record_id}: messages must contain exactly three entries")
        return errors

    roles = [message.get("role") for message in messages]
    if roles != EXPECTED_ROLES:
        errors.append(f"{record_id}: incorrect chat roles {roles}")

    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{record_id}: empty content in message {message_index}")

    try:
        target = json.loads(messages[2]["content"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        errors.append(f"{record_id}: assistant output is not valid JSON: {error}")
        return errors

    if not isinstance(target, dict):
        errors.append(f"{record_id}: assistant output must be a JSON object")
        return errors

    missing = REQUIRED_TARGET_FIELDS - set(target)
    extra = set(target) - REQUIRED_TARGET_FIELDS
    if missing:
        errors.append(f"{record_id}: missing output fields {sorted(missing)}")
    if extra:
        errors.append(f"{record_id}: unexpected output fields {sorted(extra)}")

    status = target.get("status")
    if status not in EXPECTED_STATUSES:
        errors.append(f"{record_id}: unsupported status {status!r}")

    metadata = record.get("metadata", {})
    if metadata.get("expected_status") != status:
        errors.append(f"{record_id}: metadata status does not match assistant status")

    if target.get("requires_human_approval") is not True:
        errors.append(f"{record_id}: human approval must be true")

    if not isinstance(target.get("limitations"), list) or not target.get("limitations"):
        errors.append(f"{record_id}: limitations must be a non-empty list")
    if not isinstance(target.get("prohibited_automatic_actions"), list) or len(
        target.get("prohibited_automatic_actions", [])
    ) < 3:
        errors.append(f"{record_id}: at least three prohibited actions are required")
    if not str(target.get("stopping_condition", "")).strip():
        errors.append(f"{record_id}: stopping condition is required")

    source_type = record.get("source_type")
    contains_real = metadata.get("contains_real_project_measurements")
    if source_type == "real_held_out":
        if expected_split != "test":
            errors.append(f"{record_id}: real project record leaked into {expected_split}")
        if contains_real is not True:
            errors.append(f"{record_id}: real record metadata flag is incorrect")
    elif contains_real is not False:
        errors.append(f"{record_id}: synthetic record metadata flag is incorrect")

    return errors


def make_figure(records_by_split: dict[str, list[dict[str, Any]]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    statuses = sorted(EXPECTED_STATUSES)
    splits = list(FILES)
    counts = {
        split: Counter(record["metadata"]["expected_status"] for record in records)
        for split, records in records_by_split.items()
    }

    x = np.arange(len(splits))
    width = 0.24
    colours = ["#6c757d", "#2a9d8f", "#e76f51"]

    fig, axis = plt.subplots(figsize=(11, 6))
    for index, status in enumerate(statuses):
        values = [counts[split][status] for split in splits]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=status.replace("_", " ").title(),
            color=colours[index],
        )
        axis.bar_label(bars, padding=3, fontsize=9)

    axis.set_title("ClimateAdapt Instruction Dataset – Status Distribution", weight="bold")
    axis.set_ylabel("Examples")
    axis.set_xticks(x, [split.title() for split in splits])
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "instruction_dataset_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CLIMATEADAPT – INSTRUCTION DATASET QUALITY AUDIT")
    print("=" * 78)

    records_by_split = {split: read_jsonl(path) for split, path in FILES.items()}
    all_records = [
        record
        for split in FILES
        for record in records_by_split[split]
    ]

    errors: list[str] = []
    for split, records in records_by_split.items():
        for record in records:
            errors.extend(validate_record(record, split))

    ids = [record.get("id") for record in all_records]
    duplicate_ids = sorted(record_id for record_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"Duplicate record IDs: {duplicate_ids}")

    full_fingerprints: dict[str, list[str]] = defaultdict(list)
    user_fingerprints: dict[str, list[str]] = defaultdict(list)
    for record in all_records:
        full_fingerprints[fingerprint(record["messages"])].append(record["id"])
        user_fingerprints[fingerprint(record["messages"][1]["content"])].append(record["id"])

    duplicate_records = [ids for ids in full_fingerprints.values() if len(ids) > 1]
    duplicate_prompts = [ids for ids in user_fingerprints.values() if len(ids) > 1]
    if duplicate_records:
        errors.append(f"Duplicate complete conversations: {duplicate_records}")
    if duplicate_prompts:
        errors.append(f"Duplicate user prompts: {duplicate_prompts}")

    split_ids = {
        split: {record["id"] for record in records}
        for split, records in records_by_split.items()
    }
    overlap = {
        "train_validation": sorted(split_ids["train"] & split_ids["validation"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "validation_test": sorted(split_ids["validation"] & split_ids["test"]),
    }
    if any(overlap.values()):
        errors.append(f"Split ID leakage detected: {overlap}")

    real_records = [record for record in all_records if record["source_type"] == "real_held_out"]
    real_outside_test = [record["id"] for record in real_records if record["split"] != "test"]
    if real_outside_test:
        errors.append(f"Real records outside test split: {real_outside_test}")

    task_counts = Counter(record["task_type"] for record in all_records)
    status_counts = Counter(record["metadata"]["expected_status"] for record in all_records)
    source_counts = Counter(record["source_type"] for record in all_records)

    character_lengths = [
        sum(len(message["content"]) for message in record["messages"])
        for record in all_records
    ]
    approximate_tokens = [length / 4.0 for length in character_lengths]

    audit_rows: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        split_status = Counter(record["metadata"]["expected_status"] for record in records)
        audit_rows.append(
            {
                "split": split,
                "examples": len(records),
                "blocked_data_validation": split_status["blocked_data_validation"],
                "draft_ready_for_human_review": split_status["draft_ready_for_human_review"],
                "human_review_required": split_status["human_review_required"],
                "real_held_out_examples": sum(
                    record["source_type"] == "real_held_out" for record in records
                ),
            }
        )

    csv_path = OUTPUT_DIR / "instruction_dataset_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    figure_path = make_figure(records_by_split)
    summary = {
        "audit_status": "passed" if not errors else "failed",
        "total_examples": len(all_records),
        "split_counts": {split: len(records) for split, records in records_by_split.items()},
        "task_counts": dict(task_counts),
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "duplicate_record_ids": duplicate_ids,
        "duplicate_complete_conversations": duplicate_records,
        "duplicate_user_prompts": duplicate_prompts,
        "split_id_overlap": overlap,
        "real_records_outside_test": real_outside_test,
        "approximate_sequence_tokens": {
            "minimum": round(min(approximate_tokens)),
            "mean": round(sum(approximate_tokens) / len(approximate_tokens)),
            "maximum": round(max(approximate_tokens)),
            "method": "total characters divided by four; planning estimate only",
        },
        "safety_checks": {
            "human_approval_required_in_every_target": all(
                json.loads(record["messages"][2]["content"])["requires_human_approval"]
                is True
                for record in all_records
            ),
            "real_measurements_held_out": not real_outside_test,
            "invalid_input_examples_present": status_counts["blocked_data_validation"] > 0,
            "unsafe_request_task_present": task_counts["refuse_unsafe_autonomous_action"] > 0,
        },
        "errors": errors,
    }

    summary_path = OUTPUT_DIR / "instruction_dataset_audit.json"
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)

    print("\nDataset counts")
    print("-" * 78)
    for row in audit_rows:
        print(
            f"{row['split'].title():<12} {row['examples']:>4} examples | "
            f"blocked={row['blocked_data_validation']:>2} | "
            f"draft={row['draft_ready_for_human_review']:>2} | "
            f"review={row['human_review_required']:>3} | "
            f"real-held-out={row['real_held_out_examples']}"
        )

    print("\nQuality and leakage checks")
    print("-" * 78)
    print(f"Unique IDs:                  {'PASS' if not duplicate_ids else 'FAIL'}")
    print(f"Duplicate conversations:     {'PASS' if not duplicate_records else 'FAIL'}")
    print(f"Duplicate user prompts:      {'PASS' if not duplicate_prompts else 'FAIL'}")
    print(f"Split leakage:               {'PASS' if not any(overlap.values()) else 'FAIL'}")
    print(f"Real outputs held out:       {'PASS' if not real_outside_test else 'FAIL'}")
    print(f"Human approval in targets:   {'PASS' if summary['safety_checks']['human_approval_required_in_every_target'] else 'FAIL'}")
    print(
        "Approximate sequence tokens: "
        f"min={summary['approximate_sequence_tokens']['minimum']}, "
        f"mean={summary['approximate_sequence_tokens']['mean']}, "
        f"max={summary['approximate_sequence_tokens']['maximum']}"
    )

    print("\nOutputs")
    print("-" * 78)
    print("Audit JSON:", summary_path.relative_to(ROOT))
    print("Audit CSV: ", csv_path.relative_to(ROOT))
    if figure_path is not None:
        print("Figure:    ", figure_path.relative_to(ROOT))
    else:
        print("Figure:     not created because matplotlib is unavailable")

    if errors:
        print("\nAUDIT FAILED")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    print("\nRESULT: AUDIT PASSED")
    print("The dataset is ready for controlled base-model evaluation and QLoRA fine-tuning.")


if __name__ == "__main__":
    main()
