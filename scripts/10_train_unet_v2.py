from pathlib import Path
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Project paths and imports
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_FOLDER = PROJECT_ROOT / "src"

if str(SRC_FOLDER) not in sys.path:
    sys.path.insert(0, str(SRC_FOLDER))

from climateadapt.models.unet_v2 import CompactUNetV2


TILE_FOLDER = PROJECT_ROOT / "data/processed/tiles"
MODEL_FOLDER = PROJECT_ROOT / "models"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs/training_v2"
EVIDENCE_FOLDER = PROJECT_ROOT / "evidence/figures"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
EVIDENCE_FOLDER.mkdir(parents=True, exist_ok=True)


# =========================================================
# Experiment 2 configuration
# =========================================================
EXPERIMENT_NAME = "climateadapt_unet_v2"
RANDOM_SEED = 42
BATCH_SIZE = 4
MAXIMUM_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BASE_CHANNELS = 16
DROPOUT_PROBABILITY = 0.20
DECISION_THRESHOLD = 0.50
NUM_WORKERS = 0

BCE_WEIGHT = 0.50
FOCAL_TVERSKY_WEIGHT = 0.50
MAXIMUM_POSITIVE_WEIGHT = 5.0
TVERSKY_ALPHA = 0.30
TVERSKY_BETA = 0.70
TVERSKY_GAMMA = 0.75


# =========================================================
# Reproducibility
# =========================================================
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Dataset
# =========================================================
class FloodTileDataset(Dataset):
    def __init__(
        self,
        tile_folder,
        manifest,
        augment=False,
    ):
        self.tile_folder = tile_folder
        self.manifest = manifest.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        record = self.manifest.iloc[index]
        tile_path = self.tile_folder / record["file"]

        with np.load(tile_path) as tile:
            features = tile["features"].astype(np.float32)
            target = tile["target"].astype(np.float32)
            valid_mask = tile["valid_mask"].astype(np.float32)

        if self.augment:
            if np.random.random() < 0.5:
                features = np.flip(features, axis=2)
                target = np.flip(target, axis=1)
                valid_mask = np.flip(valid_mask, axis=1)

            if np.random.random() < 0.5:
                features = np.flip(features, axis=1)
                target = np.flip(target, axis=0)
                valid_mask = np.flip(valid_mask, axis=0)

        features = features.copy()
        target = target.copy()[np.newaxis, :, :]
        valid_mask = valid_mask.copy()[np.newaxis, :, :]

        return {
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid_mask),
            "row": int(record["row"]),
            "column": int(record["column"]),
        }


# =========================================================
# Masked BCE + Focal Tversky loss
# =========================================================
def masked_bce_loss(
    logits,
    targets,
    valid_mask,
    positive_weight,
):
    pixel_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction="none",
    )

    return (
        (pixel_loss * valid_mask).sum()
        / valid_mask.sum().clamp_min(1.0)
    )


def masked_focal_tversky_loss(
    logits,
    targets,
    valid_mask,
    alpha=TVERSKY_ALPHA,
    beta=TVERSKY_BETA,
    gamma=TVERSKY_GAMMA,
    smooth=1.0,
):
    probabilities = torch.sigmoid(logits) * valid_mask
    targets = targets * valid_mask

    true_positive = (probabilities * targets).sum()
    false_positive = (
        probabilities * (1.0 - targets)
    ).sum()
    false_negative = (
        (1.0 - probabilities) * targets
    ).sum()

    tversky_index = (
        (true_positive + smooth)
        / (
            true_positive
            + alpha * false_positive
            + beta * false_negative
            + smooth
        )
    )

    return torch.pow(
        1.0 - tversky_index,
        gamma,
    )


def combined_loss(
    logits,
    targets,
    valid_mask,
    positive_weight,
):
    bce = masked_bce_loss(
        logits,
        targets,
        valid_mask,
        positive_weight,
    )
    focal_tversky = masked_focal_tversky_loss(
        logits,
        targets,
        valid_mask,
    )

    total = (
        BCE_WEIGHT * bce
        + FOCAL_TVERSKY_WEIGHT * focal_tversky
    )

    return total, bce, focal_tversky


# =========================================================
# Metrics
# =========================================================
def calculate_metrics_from_arrays(
    targets,
    probabilities,
    threshold=DECISION_THRESHOLD,
):
    targets = np.asarray(targets) >= 0.5
    predictions = np.asarray(probabilities) >= threshold

    true_positive = int((predictions & targets).sum())
    true_negative = int((~predictions & ~targets).sum())
    false_positive = int((predictions & ~targets).sum())
    false_negative = int((~predictions & targets).sum())

    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    dice = (
        2.0 * true_positive
        / (2.0 * true_positive + false_positive + false_negative)
        if 2.0 * true_positive + false_positive + false_negative > 0
        else 0.0
    )
    iou = (
        true_positive
        / (true_positive + false_positive + false_negative)
        if true_positive + false_positive + false_negative > 0
        else 0.0
    )

    return {
        "accuracy": (
            (true_positive + true_negative) / total
            if total > 0
            else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "specificity": (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive > 0
            else 0.0
        ),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def average_precision_score_numpy(targets, probabilities):
    """Non-interpolated average precision, equivalent to PR-AUC/AP."""
    targets = np.asarray(targets, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    positive_count = int(targets.sum())

    if positive_count == 0:
        raise ValueError(
            "Validation data contains no positive pixels, so PR-AUC "
            "cannot be calculated."
        )

    order = np.argsort(
        -probabilities,
        kind="mergesort",
    )
    sorted_targets = targets[order]
    cumulative_positive = np.cumsum(sorted_targets)
    ranks = np.arange(1, len(sorted_targets) + 1)
    precision_at_rank = cumulative_positive / ranks

    return float(
        precision_at_rank[sorted_targets == 1].sum()
        / positive_count
    )


def merge_overlapping_predictions(
    global_indices,
    targets,
    probabilities,
):
    global_indices = np.concatenate(global_indices)
    targets = np.concatenate(targets).astype(np.uint8)
    probabilities = np.concatenate(probabilities).astype(np.float64)

    order = np.argsort(global_indices, kind="mergesort")
    sorted_indices = global_indices[order]
    sorted_targets = targets[order]
    sorted_probabilities = probabilities[order]

    _, starts, counts = np.unique(
        sorted_indices,
        return_index=True,
        return_counts=True,
    )

    target_minimum = np.minimum.reduceat(
        sorted_targets,
        starts,
    )
    target_maximum = np.maximum.reduceat(
        sorted_targets,
        starts,
    )

    if not np.array_equal(target_minimum, target_maximum):
        raise ValueError(
            "Overlapping validation tiles contain conflicting targets."
        )

    merged_probabilities = (
        np.add.reduceat(sorted_probabilities, starts)
        / counts
    )

    return (
        target_maximum,
        merged_probabilities,
        int(len(global_indices) - len(starts)),
    )


# =========================================================
# Class weighting
# =========================================================
def calculate_positive_weight(dataset):
    positive_pixels = 0
    negative_pixels = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        target = sample["target"].numpy()
        mask = sample["valid_mask"].numpy()

        positive_pixels += int(
            ((target == 1) & (mask == 1)).sum()
        )
        negative_pixels += int(
            ((target == 0) & (mask == 1)).sum()
        )

    if positive_pixels == 0:
        raise ValueError("Training dataset has no positive pixels.")

    raw_weight = negative_pixels / positive_pixels
    final_weight = min(
        raw_weight,
        MAXIMUM_POSITIVE_WEIGHT,
    )

    print(f"Training positive pixels: {positive_pixels:,}")
    print(f"Training negative pixels: {negative_pixels:,}")
    print(f"Raw positive weight: {raw_weight:.3f}")
    print(f"Maximum positive weight: {MAXIMUM_POSITIVE_WEIGHT:.3f}")
    print(f"Used positive weight: {final_weight:.3f}")

    return final_weight


# =========================================================
# Training and validation epochs
# =========================================================
def run_training_epoch(
    model,
    data_loader,
    device,
    positive_weight,
    optimizer,
    gradient_scaler,
    use_mixed_precision,
):
    model.train()
    total_loss_sum = 0.0
    bce_loss_sum = 0.0
    focal_loss_sum = 0.0
    valid_pixel_sum = 0
    all_targets = []
    all_probabilities = []

    for batch in data_loader:
        features = batch["features"].to(
            device,
            non_blocking=True,
        )
        targets = batch["target"].to(
            device,
            non_blocking=True,
        )
        valid_mask = batch["valid_mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_mixed_precision,
        ):
            logits = model(features)
            total_loss, bce_loss, focal_loss = combined_loss(
                logits,
                targets,
                valid_mask,
                positive_weight,
            )

        gradient_scaler.scale(total_loss).backward()
        gradient_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        gradient_scaler.step(optimizer)
        gradient_scaler.update()

        valid_count = int(valid_mask.sum().item())
        total_loss_sum += total_loss.item() * valid_count
        bce_loss_sum += bce_loss.item() * valid_count
        focal_loss_sum += focal_loss.item() * valid_count
        valid_pixel_sum += valid_count

        probabilities = torch.sigmoid(logits.detach())
        valid = valid_mask >= 0.5
        all_targets.append(targets[valid].detach().cpu().numpy())
        all_probabilities.append(
            probabilities[valid].detach().cpu().numpy()
        )

    metrics = calculate_metrics_from_arrays(
        np.concatenate(all_targets),
        np.concatenate(all_probabilities),
    )
    metrics.update(
        {
            "loss": total_loss_sum / max(valid_pixel_sum, 1),
            "bce_loss": bce_loss_sum / max(valid_pixel_sum, 1),
            "focal_tversky_loss": (
                focal_loss_sum / max(valid_pixel_sum, 1)
            ),
        }
    )

    return metrics


def run_validation_epoch(
    model,
    data_loader,
    device,
    positive_weight,
    raster_width,
    use_mixed_precision,
):
    model.eval()
    total_loss_sum = 0.0
    bce_loss_sum = 0.0
    focal_loss_sum = 0.0
    valid_pixel_sum = 0
    global_indices = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(
                device,
                non_blocking=True,
            )
            targets = batch["target"].to(
                device,
                non_blocking=True,
            )
            valid_mask = batch["valid_mask"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_mixed_precision,
            ):
                logits = model(features)
                total_loss, bce_loss, focal_loss = combined_loss(
                    logits,
                    targets,
                    valid_mask,
                    positive_weight,
                )

            valid_count = int(valid_mask.sum().item())
            total_loss_sum += total_loss.item() * valid_count
            bce_loss_sum += bce_loss.item() * valid_count
            focal_loss_sum += focal_loss.item() * valid_count
            valid_pixel_sum += valid_count

            probabilities = torch.sigmoid(logits).cpu().numpy()
            target_arrays = targets.cpu().numpy()
            mask_arrays = valid_mask.cpu().numpy() >= 0.5
            rows = batch["row"].cpu().numpy()
            columns = batch["column"].cpu().numpy()

            for sample_index in range(len(rows)):
                sample_mask = mask_arrays[sample_index, 0]
                local_y, local_x = np.nonzero(sample_mask)

                indices = (
                    (int(rows[sample_index]) + local_y)
                    * raster_width
                    + int(columns[sample_index])
                    + local_x
                )

                global_indices.append(indices.astype(np.int64))
                all_targets.append(
                    target_arrays[
                        sample_index,
                        0,
                        local_y,
                        local_x,
                    ]
                )
                all_probabilities.append(
                    probabilities[
                        sample_index,
                        0,
                        local_y,
                        local_x,
                    ]
                )

    merged_targets, merged_probabilities, duplicate_count = (
        merge_overlapping_predictions(
            global_indices,
            all_targets,
            all_probabilities,
        )
    )

    metrics = calculate_metrics_from_arrays(
        merged_targets,
        merged_probabilities,
    )
    metrics["pr_auc"] = average_precision_score_numpy(
        merged_targets,
        merged_probabilities,
    )
    metrics["unique_pixels"] = int(len(merged_targets))
    metrics["merged_duplicate_predictions"] = duplicate_count
    metrics["loss"] = total_loss_sum / max(valid_pixel_sum, 1)
    metrics["bce_loss"] = bce_loss_sum / max(valid_pixel_sum, 1)
    metrics["focal_tversky_loss"] = (
        focal_loss_sum / max(valid_pixel_sum, 1)
    )

    return metrics


# =========================================================
# Checkpoint helper
# =========================================================
def make_checkpoint(
    model,
    epoch,
    input_channels,
    validation_metrics,
    positive_weight_value,
    tile_metadata,
):
    return {
        "experiment": "Experiment 2",
        "architecture": "CompactUNetV2_GroupNorm",
        "model_state_dict": model.state_dict(),
        "input_channels": input_channels,
        "base_channels": BASE_CHANNELS,
        "dropout_probability": DROPOUT_PROBABILITY,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "checkpoint_selection_metric": "validation_pr_auc",
        "positive_weight": positive_weight_value,
        "maximum_positive_weight": MAXIMUM_POSITIVE_WEIGHT,
        "decision_threshold_for_monitoring_only": DECISION_THRESHOLD,
        "loss_configuration": {
            "bce_weight": BCE_WEIGHT,
            "focal_tversky_weight": FOCAL_TVERSKY_WEIGHT,
            "tversky_alpha": TVERSKY_ALPHA,
            "tversky_beta": TVERSKY_BETA,
            "tversky_gamma": TVERSKY_GAMMA,
        },
        "tile_metadata": tile_metadata,
    }


# =========================================================
# Main training procedure
# =========================================================
def main():
    set_random_seeds(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_mixed_precision = device.type == "cuda"

    print("\n" + "=" * 76)
    print("CLIMATEADAPT EXPERIMENT 2 – GROUPNORM COMPACT U-NET")
    print("=" * 76)
    print(f"Device: {device}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        total_vram = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )
        print(f"GPU memory: {total_vram:.2f} GB")

    print("Mixed precision:", use_mixed_precision)
    print("Normalization layers: GroupNorm")
    print("Checkpoint metric: overlap-aware validation PR-AUC")

    metadata_path = TILE_FOLDER / "tile_metadata.json"
    manifest_path = TILE_FOLDER / "tile_manifest.csv"

    with open(metadata_path, "r", encoding="utf-8") as file:
        tile_metadata = json.load(file)

    manifest = pd.read_csv(manifest_path)
    input_channels = int(tile_metadata["number_of_channels"])

    training_manifest = manifest[
        manifest["split"] == "train"
    ].copy()
    validation_manifest = manifest[
        manifest["split"] == "validation"
    ].copy()

    if training_manifest.empty:
        raise ValueError("Training manifest is empty.")
    if validation_manifest.empty:
        raise ValueError("Validation manifest is empty.")
    if (manifest["split"] == "demonstration").any():
        raise ValueError(
            "The protected demonstration split was found in the tile "
            "manifest. Recreate tiles before training."
        )

    training_dataset = FloodTileDataset(
        TILE_FOLDER,
        training_manifest,
        augment=True,
    )
    validation_dataset = FloodTileDataset(
        TILE_FOLDER,
        validation_manifest,
        augment=False,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    raster_width = int(
        manifest["column"].max()
        + tile_metadata["tile_size_pixels"]
        + 1
    )

    print(f"Training tiles: {len(training_dataset)}")
    print(f"Validation tiles: {len(validation_dataset)}")
    print(f"Input channels: {input_channels}")

    positive_weight_value = calculate_positive_weight(
        training_dataset
    )
    positive_weight = torch.tensor(
        positive_weight_value,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 1)

    model = CompactUNetV2(
        input_channels=input_channels,
        base_channels=BASE_CHANNELS,
        dropout_probability=DROPOUT_PROBABILITY,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print(f"Model parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=0.000001,
    )
    gradient_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_mixed_precision,
    )

    best_validation_pr_auc = -1.0
    epochs_without_improvement = 0
    history_records = []
    best_model_path = (
        MODEL_FOLDER / "climateadapt_unet_v2_best.pt"
    )
    final_model_path = (
        MODEL_FOLDER / "climateadapt_unet_v2_final.pt"
    )
    start_time = time.time()

    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epoch_start = time.time()

        training_metrics = run_training_epoch(
            model,
            training_loader,
            device,
            positive_weight,
            optimizer,
            gradient_scaler,
            use_mixed_precision,
        )
        validation_metrics = run_validation_epoch(
            model,
            validation_loader,
            device,
            positive_weight,
            raster_width,
            use_mixed_precision,
        )

        scheduler.step(validation_metrics["pr_auc"])
        current_learning_rate = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start

        history_records.append(
            {
                "epoch": epoch,
                "learning_rate": current_learning_rate,
                "train_loss": training_metrics["loss"],
                "train_dice": training_metrics["dice"],
                "train_precision": training_metrics["precision"],
                "train_recall": training_metrics["recall"],
                "validation_loss": validation_metrics["loss"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "validation_dice_at_0_5": validation_metrics["dice"],
                "validation_precision_at_0_5": (
                    validation_metrics["precision"]
                ),
                "validation_recall_at_0_5": (
                    validation_metrics["recall"]
                ),
                "validation_unique_pixels": (
                    validation_metrics["unique_pixels"]
                ),
                "epoch_seconds": epoch_seconds,
            }
        )

        print(
            f"Epoch {epoch:02d}/{MAXIMUM_EPOCHS} | "
            f"train loss={training_metrics['loss']:.4f} "
            f"dice={training_metrics['dice']:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} "
            f"PR-AUC={validation_metrics['pr_auc']:.4f} "
            f"dice@0.5={validation_metrics['dice']:.4f} "
            f"P={validation_metrics['precision']:.4f} "
            f"R={validation_metrics['recall']:.4f} | "
            f"lr={current_learning_rate:.6f} | "
            f"{epoch_seconds:.1f}s"
        )

        if (
            validation_metrics["pr_auc"]
            > best_validation_pr_auc + 0.0001
        ):
            best_validation_pr_auc = validation_metrics["pr_auc"]
            epochs_without_improvement = 0

            torch.save(
                make_checkpoint(
                    model,
                    epoch,
                    input_channels,
                    validation_metrics,
                    positive_weight_value,
                    tile_metadata,
                ),
                best_model_path,
            )
            print("  Saved new best PR-AUC checkpoint.")
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping: validation PR-AUC did not "
                f"improve for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    total_training_seconds = time.time() - start_time

    torch.save(
        make_checkpoint(
            model,
            history_records[-1]["epoch"],
            input_channels,
            validation_metrics,
            positive_weight_value,
            tile_metadata,
        ),
        final_model_path,
    )

    history = pd.DataFrame(history_records)
    history_path = OUTPUT_FOLDER / "training_history_v2.csv"
    history.to_csv(history_path, index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(
        history["epoch"],
        history["train_loss"],
        label="Training",
    )
    axes[0, 0].plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation",
    )
    axes[0, 0].set_title("Combined loss")
    axes[0, 0].legend()

    axes[0, 1].plot(
        history["epoch"],
        history["validation_pr_auc"],
        color="#7b2cbf",
    )
    axes[0, 1].set_title("Overlap-aware validation PR-AUC")

    axes[1, 0].plot(
        history["epoch"],
        history["train_dice"],
        label="Training Dice",
    )
    axes[1, 0].plot(
        history["epoch"],
        history["validation_dice_at_0_5"],
        label="Validation Dice @ 0.5",
    )
    axes[1, 0].set_title("Dice monitoring metric")
    axes[1, 0].legend()

    axes[1, 1].plot(
        history["epoch"],
        history["validation_precision_at_0_5"],
        label="Precision @ 0.5",
        color="#2ca02c",
    )
    axes[1, 1].plot(
        history["epoch"],
        history["validation_recall_at_0_5"],
        label="Recall @ 0.5",
        color="#d62728",
    )
    axes[1, 1].set_title("Validation precision and recall")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)

    figure.suptitle(
        "ClimateAdapt AI – Experiment 2 U-Net Training",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    curve_path = EVIDENCE_FOLDER / "unet_v2_training_curves.png"
    figure.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("\n" + "=" * 76)
    print("EXPERIMENT 2 TRAINING COMPLETED")
    print("=" * 76)
    print(f"Best validation PR-AUC: {best_validation_pr_auc:.4f}")
    print(
        "Validation unique pixels per epoch: "
        f"{validation_metrics['unique_pixels']:,}"
    )
    print(
        "Overlapping duplicate predictions merged: "
        f"{validation_metrics['merged_duplicate_predictions']:,}"
    )
    print(
        f"Training time: {total_training_seconds / 60:.2f} minutes"
    )
    print(f"Best model: {best_model_path.relative_to(PROJECT_ROOT)}")
    print(f"Final model: {final_model_path.relative_to(PROJECT_ROOT)}")
    print(f"History: {history_path.relative_to(PROJECT_ROOT)}")
    print(f"Training curves: {curve_path.relative_to(PROJECT_ROOT)}")
    print("=" * 76)
    print("The test and Waimea/Monaco demonstration data were not used.")


if __name__ == "__main__":
    main()
