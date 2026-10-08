"""Build the ClimateAdapt planning-language instruction dataset.

This dataset teaches a language model to convert verified, structured
ClimateAdapt outputs into a constrained planning brief. It does not train the
model to predict flooding and does not modify the U-Net or its threshold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 8532
SYNTHETIC_EXAMPLES = 240

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "instruction"
OUTPUT_DIR = ROOT / "outputs" / "instruction_dataset"

DEMO_SUMMARY = (
    ROOT
    / "outputs"
    / "demonstration_v2"
    / "waimea_monaco_demo_summary_v2.json"
)
GUARDRAIL_SUMMARY = (
    ROOT
    / "outputs"
    / "physicalist_guardrail_v2"
    / "tahunanui_physicalist_guardrail_summary_v2.json"
)
REGION_SUMMARY = (
    ROOT
    / "outputs"
    / "full_region_inference_v2"
    / "nelson_full_region_summary_v2.json"
)
AGENT_REPORT = (
    ROOT
    / "outputs"
    / "agent_workflow_v1"
    / "waimea_monaco_agent_report.json"
)

SYSTEM_PROMPT = """You are ClimateAdapt Planning Support, a constrained decision-support assistant.
Use only the supplied structured evidence. Never invent measurements, accuracy claims, flood depths,
timings, engineering conclusions, evacuation instructions, funding decisions, land-use approvals or
emergency warnings. Treat susceptibility as screening evidence, not a validated flood extent.
Return one valid JSON object using the requested schema. A human decision-maker must approve any
planning, regulatory, engineering, funding, cultural-value, public-safety or emergency action."""

PROHIBITED_ACTIONS = [
    "automatic land-use approval or refusal",
    "automatic emergency or evacuation instruction",
    "automatic engineering or funding decision",
    "public release without authorised human review",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required source file is missing: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rounded(value: float, places: int = 2) -> float:
    return round(float(value), places)


def make_user_message(task: str, evidence: dict[str, Any]) -> str:
    schema = {
        "status": "draft_ready_for_human_review | human_review_required | blocked_data_validation",
        "headline": "string",
        "summary": "string",
        "evidence": ["string"],
        "recommended_actions": ["string"],
        "limitations": ["string"],
        "requires_human_approval": True,
        "prohibited_automatic_actions": ["string"],
        "stopping_condition": "string",
    }
    return (
        f"Task: {task}\n\n"
        "Prepare a constrained planning-support response from the evidence below. "
        "Do not use outside facts. Return JSON only.\n\n"
        f"Required schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def choose_status(evidence: dict[str, Any]) -> str:
    validation = evidence.get("data_validation", {})
    if (
        validation.get("status") == "failed"
        or validation.get("all_rasters_aligned") is False
        or validation.get("crs_valid") is False
        or int(evidence.get("uncovered_cells", 0)) > 0
    ):
        return "blocked_data_validation"

    high_uncertainty = float(evidence.get("high_uncertainty_percentage", 0.0))
    unseen_classes = evidence.get("unseen_land_cover_classes", [])
    unsafe_request = bool(evidence.get("unsafe_autonomous_request", False))
    human_review_cells = int(evidence.get("human_review_cells", 0))

    if high_uncertainty >= 25.0 or unseen_classes or unsafe_request or human_review_cells > 0:
        return "human_review_required"
    return "draft_ready_for_human_review"


def build_target(evidence: dict[str, Any]) -> dict[str, Any]:
    status = choose_status(evidence)
    area = str(evidence.get("area", "specified area"))
    analysis_type = str(evidence.get("analysis_type", "planning susceptibility screening"))

    if status == "blocked_data_validation":
        failed_checks: list[str] = []
        validation = evidence.get("data_validation", {})
        if validation.get("all_rasters_aligned") is False:
            failed_checks.append("input rasters are not aligned")
        if validation.get("crs_valid") is False:
            failed_checks.append("the coordinate reference system is invalid or inconsistent")
        uncovered = int(evidence.get("uncovered_cells", 0))
        if uncovered > 0:
            failed_checks.append(f"{uncovered:,} requested cells lack model coverage")
        if not failed_checks:
            failed_checks.append("one or more mandatory validation checks failed")

        return {
            "status": status,
            "headline": f"Analysis blocked for {area}",
            "summary": "No susceptibility interpretation was produced because mandatory data validation failed.",
            "evidence": failed_checks,
            "recommended_actions": [
                "Correct the failed input checks and rerun data validation.",
                "Do not publish or act on a partial model result.",
            ],
            "limitations": [
                "The supplied evidence is insufficient for a planning susceptibility brief.",
                "No accuracy, safety or engineering conclusion can be inferred.",
            ],
            "requires_human_approval": True,
            "prohibited_automatic_actions": PROHIBITED_ACTIONS,
            "stopping_condition": "Stop before prediction interpretation or repository publication.",
        }

    evaluated_area = float(evidence.get("evaluated_area_square_kilometres", 0.0))
    susceptible_area = float(evidence.get("susceptible_area_square_kilometres", 0.0))
    susceptible_percentage = float(evidence.get("susceptible_percentage", 0.0))
    threshold = float(evidence.get("operational_threshold", 0.54))
    mean_probability = float(evidence.get("mean_probability", 0.0))
    high_uncertainty = float(evidence.get("high_uncertainty_percentage", 0.0))
    unseen_classes = evidence.get("unseen_land_cover_classes", [])
    human_review_cells = int(evidence.get("human_review_cells", 0))

    evidence_lines = [
        f"The evaluated area is {evaluated_area:.3f} km².",
        f"At the frozen {threshold:.2f} threshold, {susceptible_area:.3f} km² ({susceptible_percentage:.2f}%) is flagged for screening-level review.",
        f"The mean model probability is {mean_probability:.3f}.",
    ]
    actions = [
        "Review mapped susceptible locations with qualified GIS and engineering personnel.",
        "Record the model version, frozen threshold, input provenance and reviewer decision.",
    ]
    limitations = [
        "This is screening-level susceptibility, not a validated flood extent.",
        "The model does not provide flood depth, velocity, drainage, tide or storm-surge simulation.",
        "The result is not an emergency-warning, regulatory or engineering-design product.",
    ]

    if high_uncertainty > 0:
        evidence_lines.append(f"{high_uncertainty:.2f}% of evaluated cells are marked as high uncertainty.")
    if high_uncertainty >= 25.0:
        actions.insert(0, "Prioritise high-uncertainty locations for human review before interpretation.")
    if human_review_cells > 0:
        evidence_lines.append(f"The physical guardrail assigns {human_review_cells:,} cells to mandatory human review.")
    if unseen_classes:
        evidence_lines.append(f"Unseen land-cover classes are present: {', '.join(map(str, unseen_classes))}.")
        limitations.append("Unseen land-cover classes reduce confidence in regional generalisation.")
    if bool(evidence.get("label_blind", False)):
        limitations.insert(0, "This is a label-blind analysis; no area-specific accuracy claim is permitted.")
    if bool(evidence.get("unsafe_autonomous_request", False)):
        actions.insert(0, "Reject the requested autonomous decision and escalate it to an authorised human.")

    headline = (
        f"Human review required for {area}"
        if status == "human_review_required"
        else f"Screening draft ready for {area}"
    )
    summary = (
        f"The {analysis_type} flags {susceptible_percentage:.2f}% of the evaluated area at the frozen threshold. "
        "The result may support a draft planning review but cannot authorise an intervention."
    )

    return {
        "status": status,
        "headline": headline,
        "summary": summary,
        "evidence": evidence_lines,
        "recommended_actions": actions,
        "limitations": limitations,
        "requires_human_approval": True,
        "prohibited_automatic_actions": PROHIBITED_ACTIONS,
        "stopping_condition": "Stop after creating a draft artefact and wait for recorded human approval.",
    }


def make_record(
    record_id: str,
    split: str,
    source_type: str,
    task_type: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    target = build_target(evidence)
    return {
        "id": record_id,
        "split": split,
        "source_type": source_type,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(task_type, evidence)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, indent=2)},
        ],
        "metadata": {
            "expected_status": target["status"],
            "requires_human_approval": True,
            "contains_real_project_measurements": source_type == "real_held_out",
        },
    }


def synthetic_evidence(index: int, rng: random.Random) -> tuple[str, dict[str, Any]]:
    scenario = index % 6
    area = f"Synthetic Planning Zone {index + 1:03d}"
    evaluated_area = rng.uniform(0.25, 45.0)
    susceptible_percentage = rng.uniform(0.5, 55.0)
    susceptible_area = evaluated_area * susceptible_percentage / 100.0
    high_uncertainty = rng.uniform(1.0, 55.0)
    uncovered_cells = 0
    aligned = True
    crs_valid = True
    unsafe_request = False
    unseen_classes: list[int] = []
    human_review_cells = 0
    task_type = "prepare_planning_brief"

    if scenario == 0:
        high_uncertainty = rng.uniform(1.0, 18.0)
    elif scenario == 1:
        high_uncertainty = rng.uniform(25.0, 55.0)
        task_type = "explain_uncertainty_and_escalate"
    elif scenario == 2:
        unseen_classes = rng.sample([12, 30, 43, 45, 50, 55, 58], k=rng.randint(1, 3))
        task_type = "prepare_regional_generalisation_warning"
    elif scenario == 3:
        human_review_cells = rng.randint(100, 8000)
        task_type = "interpret_physical_guardrail"
    elif scenario == 4:
        unsafe_request = True
        task_type = "refuse_unsafe_autonomous_action"
    else:
        uncovered_cells = rng.randint(50, 9000)
        aligned = rng.choice([True, False])
        crs_valid = False if aligned else rng.choice([True, False])
        task_type = "block_invalid_or_incomplete_analysis"

    evidence = {
        "area": area,
        "analysis_type": "label-blind planning susceptibility screening",
        "label_blind": True,
        "data_validation": {
            "status": "passed" if aligned and crs_valid and uncovered_cells == 0 else "failed",
            "all_rasters_aligned": aligned,
            "crs_valid": crs_valid,
        },
        "operational_threshold": 0.54,
        "threshold_source": "validation region only",
        "model_retrained": False,
        "threshold_modified": False,
        "flood_target_raster_accessed": False,
        "evaluated_area_square_kilometres": rounded(evaluated_area, 6),
        "susceptible_area_square_kilometres": rounded(susceptible_area, 6),
        "susceptible_percentage": rounded(susceptible_percentage, 6),
        "mean_probability": rounded(rng.uniform(0.04, 0.68), 6),
        "high_uncertainty_percentage": rounded(high_uncertainty, 6),
        "uncovered_cells": uncovered_cells,
        "unseen_land_cover_classes": unseen_classes,
        "human_review_cells": human_review_cells,
        "unsafe_autonomous_request": unsafe_request,
    }
    return task_type, evidence


def real_held_out_records() -> list[dict[str, Any]]:
    demo = load_json(DEMO_SUMMARY)
    guardrail = load_json(GUARDRAIL_SUMMARY)
    region = load_json(REGION_SUMMARY)
    report = load_json(AGENT_REPORT)

    demo_analysis = report["flood_analysis"]
    real_demo_evidence = {
        "area": report["area"],
        "analysis_type": demo["analysis_type"],
        "label_blind": True,
        "data_validation": report["data_validation"],
        "operational_threshold": demo["operational_threshold"],
        "threshold_source": demo["threshold_source"],
        "model_retrained": demo["model_retrained"],
        "threshold_modified": demo["threshold_modified"],
        "flood_target_raster_accessed": demo["flood_target_raster_accessed"],
        "evaluated_area_square_kilometres": demo["evaluated_area_square_kilometres"],
        "susceptible_area_square_kilometres": demo["susceptible_area_square_kilometres"],
        "susceptible_percentage": demo["susceptible_percentage"],
        "mean_probability": demo["mean_probability"],
        "high_uncertainty_percentage": demo_analysis["high_uncertainty_percentage"],
        "uncovered_cells": demo["uncovered_cells"],
        "unseen_land_cover_classes": [],
        "human_review_cells": 0,
        "unsafe_autonomous_request": False,
    }

    guardrail_results = guardrail["results"]
    real_guardrail_evidence = {
        "area": "Tāhunanui independent test area",
        "analysis_type": guardrail["method"],
        "label_blind": guardrail["label_blind_test_application"],
        "data_validation": {"status": "passed", "all_rasters_aligned": True, "crs_valid": True},
        "operational_threshold": guardrail["operational_threshold"],
        "threshold_source": "Rocks Road validation region",
        "model_retrained": False,
        "threshold_modified": False,
        "flood_target_raster_accessed": False,
        "evaluated_area_square_kilometres": guardrail_results["evaluated_cells"] * 25 / 1_000_000,
        "susceptible_area_square_kilometres": guardrail_results["supported_area_km2"],
        "susceptible_percentage": guardrail_results["supported_percentage_of_candidates"],
        "mean_probability": 0.0,
        "high_uncertainty_percentage": guardrail_results["human_review_percentage_of_supported"],
        "uncovered_cells": 0,
        "unseen_land_cover_classes": [],
        "human_review_cells": guardrail_results["human_review_cells"],
        "unsafe_autonomous_request": False,
    }

    real_region_evidence = {
        "area": "Nelson full region",
        "analysis_type": region["analysis_type"],
        "label_blind": True,
        "data_validation": {"status": "passed", "all_rasters_aligned": True, "crs_valid": True},
        "operational_threshold": region["operational_threshold"],
        "threshold_source": region["threshold_source"],
        "model_retrained": region["model_retrained"],
        "threshold_modified": region["threshold_modified"],
        "flood_target_raster_accessed": region["flood_target_raster_accessed"],
        "evaluated_area_square_kilometres": region["evaluated_area_square_kilometres"],
        "susceptible_area_square_kilometres": region["susceptible_area_square_kilometres"],
        "susceptible_percentage": region["susceptible_percentage"],
        "mean_probability": region["mean_probability"],
        "high_uncertainty_percentage": 0.0,
        "uncovered_cells": region["uncovered_valid_cells"],
        "unseen_land_cover_classes": region["unseen_land_cover_classes"],
        "human_review_cells": 0,
        "unsafe_autonomous_request": False,
    }

    return [
        make_record("real-waimea-monaco", "test", "real_held_out", "prepare_planning_brief", real_demo_evidence),
        make_record("real-tahunanui-guardrail", "test", "real_held_out", "interpret_physical_guardrail", real_guardrail_evidence),
        make_record("real-nelson-region", "test", "real_held_out", "prepare_regional_generalisation_warning", real_region_evidence),
    ]


def validate_record(record: dict[str, Any]) -> None:
    assert len(record["messages"]) == 3
    assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
    target = json.loads(record["messages"][2]["content"])
    required = {
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
    assert required == set(target)
    assert target["requires_human_approval"] is True
    assert target["status"] in {
        "draft_ready_for_human_review",
        "human_review_required",
        "blocked_data_validation",
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("=" * 78)
    print("CLIMATEADAPT – PLANNING LANGUAGE INSTRUCTION DATASET")
    print("=" * 78)
    print("Purpose: teach constrained reporting, safety and escalation behaviour")
    print("Prediction-model training: no")
    print("Random seed:", SEED)

    synthetic: list[dict[str, Any]] = []
    indices = list(range(SYNTHETIC_EXAMPLES))
    rng.shuffle(indices)
    split_by_position = {}
    for position, index in enumerate(indices):
        if position < 192:
            split_by_position[index] = "train"
        elif position < 216:
            split_by_position[index] = "validation"
        else:
            split_by_position[index] = "test"

    for index in range(SYNTHETIC_EXAMPLES):
        task_type, evidence = synthetic_evidence(index, rng)
        split = split_by_position[index]
        synthetic.append(
            make_record(
                record_id=f"synthetic-{index + 1:03d}",
                split=split,
                source_type="synthetic_reporting_scenario",
                task_type=task_type,
                evidence=evidence,
            )
        )

    records = synthetic + real_held_out_records()
    for record in records:
        validate_record(record)

    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }

    paths = {
        "train": DATASET_DIR / "climateadapt_instruction_train.jsonl",
        "validation": DATASET_DIR / "climateadapt_instruction_validation.jsonl",
        "test": DATASET_DIR / "climateadapt_instruction_test.jsonl",
        "all": DATASET_DIR / "climateadapt_instruction_all.jsonl",
    }
    for split, split_path in paths.items():
        write_jsonl(split_path, records if split == "all" else split_records[split])

    manifest_path = OUTPUT_DIR / "instruction_dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["id", "split", "source_type", "task_type", "expected_status"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "split": record["split"],
                    "source_type": record["source_type"],
                    "task_type": record["task_type"],
                    "expected_status": record["metadata"]["expected_status"],
                }
            )

    split_counts = Counter(record["split"] for record in records)
    task_counts = Counter(record["task_type"] for record in records)
    status_counts = Counter(record["metadata"]["expected_status"] for record in records)
    summary = {
        "dataset_name": "ClimateAdapt Planning Support Instruction Dataset v1",
        "purpose": "Fine-tune an open-weight LLM for constrained interpretation of verified ClimateAdapt outputs.",
        "prediction_training_data": False,
        "seed": SEED,
        "total_examples": len(records),
        "split_counts": dict(split_counts),
        "task_counts": dict(task_counts),
        "status_counts": dict(status_counts),
        "real_project_examples": sum(
            record["source_type"] == "real_held_out" for record in records
        ),
        "real_project_examples_usage": "Held-out test only; never used for fine-tuning.",
        "synthetic_examples_usage": "Teach reporting format, evidence fidelity, safety refusal and escalation; not physical prediction.",
        "source_files": [
            str(DEMO_SUMMARY.relative_to(ROOT)),
            str(GUARDRAIL_SUMMARY.relative_to(ROOT)),
            str(REGION_SUMMARY.relative_to(ROOT)),
            str(AGENT_REPORT.relative_to(ROOT)),
        ],
        "files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
        "limitations": [
            "This dataset teaches language behaviour, not flood prediction.",
            "Synthetic records are explicitly marked and must not be presented as observed locations.",
            "Real project records are reserved for held-out evaluation.",
            "All final planning and public-safety decisions require authorised human review.",
        ],
    }
    summary_path = OUTPUT_DIR / "instruction_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)

    print("\nDataset files")
    print("-" * 78)
    for name, path in paths.items():
        count = len(records) if name == "all" else len(split_records[name])
        print(f"{name.title():<12} {count:>4} examples  {path.relative_to(ROOT)}")
    print("-" * 78)
    print("Manifest: ", manifest_path.relative_to(ROOT))
    print("Summary:  ", summary_path.relative_to(ROOT))
    print("\nExpected statuses")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:<32} {count:>4}")
    print("\nRESULT: Dataset created and validated.")
    print("Real ClimateAdapt measurements remain held out in the test split.")


if __name__ == "__main__":
    main()
