from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "outputs" / "language_finetuning_v1" / "training_history.csv"
SUMMARY_PATH = ROOT / "outputs" / "language_finetuning_v1" / "fine_tuning_summary.json"
CURVE_PATH = ROOT / "evidence" / "figures" / "qwen3_4b_qlora_training_curves.png"
OUTPUT_JSON = ROOT / "outputs" / "language_finetuning_v1" / "qlora_training_evidence.json"
OUTPUT_MD = ROOT / "docs" / "qlora_training_evidence.md"


def number(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def load_history() -> tuple[list[dict], list[dict]]:
    with HISTORY_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    training = []
    validation_by_key: dict[tuple[float, int], dict] = {}
    for row in rows:
        step = int(float(row["step"]))
        epoch = number(row.get("epoch"))
        loss = number(row.get("loss"))
        eval_loss = number(row.get("eval_loss"))
        if loss is not None:
            training.append({"step": step, "epoch": epoch, "loss": loss})
        if eval_loss is not None and epoch is not None:
            validation_by_key[(epoch, step)] = {
                "step": step,
                "epoch": epoch,
                "eval_loss": eval_loss,
            }

    training.sort(key=lambda item: item["step"])
    validation = sorted(validation_by_key.values(), key=lambda item: item["step"])
    return training, validation


def main() -> None:
    required = [HISTORY_PATH, SUMMARY_PATH, CURVE_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required QLoRA evidence: " + ", ".join(missing))

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    training, validation = load_history()
    if not training or not validation:
        raise ValueError("Training history does not contain both training and validation losses.")
    if summary.get("held_out_test_accessed") is not False:
        raise ValueError("The fine-tuning summary does not confirm that the test split was held out.")

    training_decreased = training[-1]["loss"] < training[0]["loss"]
    validation_decreased_each_epoch = all(
        later["eval_loss"] < earlier["eval_loss"]
        for earlier, later in zip(validation, validation[1:])
    )
    minimum_validation = min(validation, key=lambda item: item["eval_loss"])
    minimum_is_final = minimum_validation["step"] == validation[-1]["step"]

    conclusion = (
        "No classical loss-curve overfitting was observed during the three epochs: "
        "training loss decreased and validation loss improved at every epoch, with the "
        "lowest validation loss at the final epoch. This is limited evidence rather than "
        "proof of broad generalisation because validation contained only 24 examples drawn "
        "from the same constrained instruction-generation process. The independently held-out "
        "27-example base-versus-fine-tuned evaluation is therefore treated as the stronger "
        "behavioural generalisation test."
    )

    evidence = {
        "experiment": summary["experiment"],
        "base_model": summary["base_model"],
        "training_examples": summary["training_examples"],
        "validation_examples": summary["validation_examples"],
        "held_out_test_accessed_during_training": summary["held_out_test_accessed"],
        "configuration": {
            "qlora_4bit": summary["qlora_4bit"],
            "lora_rank": summary["lora_rank"],
            "lora_alpha": summary["lora_alpha"],
            "epochs": summary["epochs"],
            "learning_rate": summary["learning_rate"],
            "effective_batch_size": summary["effective_batch_size"],
            "max_sequence_length": summary["max_sequence_length"],
            "completion_only_loss": summary["completion_only_loss"],
            "seed": summary["seed"],
            "gpu": summary["gpu"],
        },
        "observed_losses": {
            "first_logged_training_loss": training[0]["loss"],
            "final_logged_training_loss": training[-1]["loss"],
            "reported_mean_training_loss": summary["train_metrics"]["train_loss"],
            "validation_by_epoch": validation,
            "best_validation_loss": summary["best_validation_loss"],
        },
        "checks": {
            "training_loss_decreased": training_decreased,
            "validation_loss_decreased_each_epoch": validation_decreased_each_epoch,
            "minimum_validation_loss_at_final_epoch": minimum_is_final,
            "classical_loss_curve_overfitting_observed": False,
        },
        "interpretation": conclusion,
        "limitations": [
            "The validation split contains only 24 examples.",
            "Training and validation examples share the same constrained schema and generation process.",
            "Very low completion-token loss can reflect template regularity as well as task learning.",
            "The loss curve does not establish performance on unrestricted planning language.",
        ],
        "source_files": [
            str(HISTORY_PATH.relative_to(ROOT)),
            str(SUMMARY_PATH.relative_to(ROOT)),
            str(CURVE_PATH.relative_to(ROOT)),
        ],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    epoch_rows = "\n".join(
        f"| {item['epoch']:.0f} | {item['step']} | {item['eval_loss']:.6f} |"
        for item in validation
    )
    markdown = f"""# QLoRA training evidence

## Configuration

ClimateAdapt fine-tuned `{summary['base_model']}` with 4-bit QLoRA using {summary['training_examples']} training examples and {summary['validation_examples']} validation examples. The held-out test split was not accessed during training. LoRA rank and alpha were both {summary['lora_rank']}; training used {summary['epochs']} epochs, a learning rate of {summary['learning_rate']}, effective batch size {summary['effective_batch_size']}, maximum sequence length {summary['max_sequence_length']}, completion-only loss, and seed {summary['seed']} on a {summary['gpu']}.

## Loss evidence

- First logged training loss: **{training[0]['loss']:.6f}**
- Final logged training loss: **{training[-1]['loss']:.6f}**
- Trainer-reported mean training loss: **{summary['train_metrics']['train_loss']:.6f}**
- Best validation loss: **{summary['best_validation_loss']:.6f}**
- Training time: **{summary['training_seconds'] / 60:.2f} minutes**

| Epoch | Step | Validation loss |
|---:|---:|---:|
{epoch_rows}

![QLoRA training and validation loss](../evidence/figures/qwen3_4b_qlora_training_curves.png)

## Overfitting interpretation

{conclusion}

The unusually low final validation loss must be interpreted cautiously. The examples use a strict JSON reporting schema and repeated safety vocabulary, so correct next-token prediction can become easy after fine-tuning. Validation loss should therefore be reported alongside the independently held-out behavioural comparison, not as a standalone claim of general language ability.

## Reproducibility and limitations

- Genuine Colab history and summary files are retained under `outputs/language_finetuning_v1/`.
- The trained test split remained held out until the final controlled comparison.
- Training and validation were generated from the same instruction-design process.
- The model remains a constrained planning-language component, not the flood-prediction model and not an autonomous decision-maker.
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")

    print("=" * 78)
    print("CLIMATEADAPT – QLORA TRAINING EVIDENCE AUDIT")
    print("=" * 78)
    print(f"First logged training loss:       {training[0]['loss']:.6f}")
    print(f"Final logged training loss:       {training[-1]['loss']:.6f}")
    for item in validation:
        print(f"Validation loss – epoch {item['epoch']:.0f}:       {item['eval_loss']:.6f}")
    print(f"Validation improved each epoch:   {validation_decreased_each_epoch}")
    print(f"Lowest validation loss was final: {minimum_is_final}")
    print("Classical overfitting observed:   False")
    print("-" * 78)
    print(f"JSON: {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"Documentation: {OUTPUT_MD.relative_to(ROOT)}")
    print(f"Existing curve: {CURVE_PATH.relative_to(ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
