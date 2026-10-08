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

from climateadapt.models.unet import CompactUNet


TILE_FOLDER = PROJECT_ROOT / "data/processed/tiles"
MODEL_FOLDER = PROJECT_ROOT / "models"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs/training"
EVIDENCE_FOLDER = PROJECT_ROOT / "evidence/figures"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
EVIDENCE_FOLDER.mkdir(parents=True, exist_ok=True)


# =========================================================
# Training configuration
# =========================================================
RANDOM_SEED = 42
BATCH_SIZE = 4
MAXIMUM_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BASE_CHANNELS = 16
DROPOUT_PROBABILITY = 0.20
DECISION_THRESHOLD = 0.50
NUM_WORKERS = 0

BCE_WEIGHT = 0.50
DICE_WEIGHT = 0.50
MAXIMUM_POSITIVE_WEIGHT = 20.0


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
        self.manifest = manifest.reset_index(
            drop=True
        )
        self.augment = augment

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        record = self.manifest.iloc[index]
        tile_path = (
            self.tile_folder
            / record["file"]
        )

        with np.load(tile_path) as tile:
            features = tile["features"].astype(
                np.float32
            )
            target = tile["target"].astype(
                np.float32
            )
            valid_mask = tile[
                "valid_mask"
            ].astype(np.float32)

        if self.augment:
            if np.random.random() < 0.5:
                features = np.flip(
                    features,
                    axis=2,
                )
                target = np.flip(
                    target,
                    axis=1,
                )
                valid_mask = np.flip(
                    valid_mask,
                    axis=1,
                )

            if np.random.random() < 0.5:
                features = np.flip(
                    features,
                    axis=1,
                )
                target = np.flip(
                    target,
                    axis=0,
                )
                valid_mask = np.flip(
                    valid_mask,
                    axis=0,
                )

        # np.flip creates negative strides, so copy arrays
        features = features.copy()
        target = target.copy()
        valid_mask = valid_mask.copy()

        target = target[np.newaxis, :, :]
        valid_mask = valid_mask[
            np.newaxis,
            :,
            :,
        ]

        return {
            "features": torch.from_numpy(
                features
            ),
            "target": torch.from_numpy(
                target
            ),
            "valid_mask": torch.from_numpy(
                valid_mask
            ),
        }


# =========================================================
# Masked BCE + Dice loss
# =========================================================
def masked_bce_loss(
    logits,
    targets,
    valid_mask,
    positive_weight,
):
    pixel_loss = (
        F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=positive_weight,
            reduction="none",
        )
    )

    masked_loss = pixel_loss * valid_mask

    return (
        masked_loss.sum()
        / valid_mask.sum().clamp_min(1.0)
    )


def masked_dice_loss(
    logits,
    targets,
    valid_mask,
    smooth=1.0,
):
    probabilities = torch.sigmoid(
        logits
    )

    probabilities = (
        probabilities * valid_mask
    )

    targets = targets * valid_mask

    intersection = (
        probabilities * targets
    ).sum()

    denominator = (
        probabilities.sum()
        + targets.sum()
    )

    dice_score = (
        (2.0 * intersection + smooth)
        / (denominator + smooth)
    )

    return 1.0 - dice_score


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

    dice = masked_dice_loss(
        logits,
        targets,
        valid_mask,
    )

    total = (
        BCE_WEIGHT * bce
        + DICE_WEIGHT * dice
    )

    return total, bce, dice


# =========================================================
# Metrics
# =========================================================
def update_confusion_counts(
    logits,
    targets,
    valid_mask,
    threshold,
):
    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    )

    targets_binary = targets >= 0.5
    valid_binary = valid_mask >= 0.5

    true_positive = (
        predictions
        & targets_binary
        & valid_binary
    ).sum().item()

    true_negative = (
        (~predictions)
        & (~targets_binary)
        & valid_binary
    ).sum().item()

    false_positive = (
        predictions
        & (~targets_binary)
        & valid_binary
    ).sum().item()

    false_negative = (
        (~predictions)
        & targets_binary
        & valid_binary
    ).sum().item()

    return {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def calculate_metrics(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    intersection_over_union = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    dice_score = (
        (2 * tp)
        / ((2 * tp) + fp + fn)
        if ((2 * tp) + fp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "iou": intersection_over_union,
        "dice": dice_score,
        "specificity": specificity,
    }


# =========================================================
# Class-weight calculation
# =========================================================
def calculate_positive_weight(
    dataset,
):
    positive_pixels = 0
    negative_pixels = 0

    for index in range(len(dataset)):
        sample = dataset[index]

        target = sample["target"].numpy()
        mask = sample[
            "valid_mask"
        ].numpy()

        positive_pixels += int(
            (
                (target == 1)
                & (mask == 1)
            ).sum()
        )

        negative_pixels += int(
            (
                (target == 0)
                & (mask == 1)
            ).sum()
        )

    if positive_pixels == 0:
        raise ValueError(
            "Training dataset has no positive pixels."
        )

    raw_weight = (
        negative_pixels / positive_pixels
    )

    final_weight = min(
        raw_weight,
        MAXIMUM_POSITIVE_WEIGHT,
    )

    print(
        f"Training positive pixels: "
        f"{positive_pixels:,}"
    )

    print(
        f"Training negative pixels: "
        f"{negative_pixels:,}"
    )

    print(
        f"Raw positive weight: "
        f"{raw_weight:.3f}"
    )

    print(
        f"Used positive weight: "
        f"{final_weight:.3f}"
    )

    return final_weight


# =========================================================
# Run one epoch
# =========================================================
def run_epoch(
    model,
    data_loader,
    device,
    positive_weight,
    optimizer=None,
    gradient_scaler=None,
    use_mixed_precision=False,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_total_loss = 0.0
    running_bce_loss = 0.0
    running_dice_loss = 0.0
    batch_count = 0

    confusion_counts = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }

    for batch in data_loader:
        features = batch[
            "features"
        ].to(
            device,
            non_blocking=True,
        )

        targets = batch[
            "target"
        ].to(
            device,
            non_blocking=True,
        )

        valid_mask = batch[
            "valid_mask"
        ].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_mixed_precision,
            ):
                logits = model(features)

                total_loss, bce_loss, dice_loss = (
                    combined_loss(
                        logits,
                        targets,
                        valid_mask,
                        positive_weight,
                    )
                )

            if training:
                gradient_scaler.scale(
                    total_loss
                ).backward()

                gradient_scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                gradient_scaler.step(
                    optimizer
                )

                gradient_scaler.update()

        counts = update_confusion_counts(
            logits.detach(),
            targets,
            valid_mask,
            DECISION_THRESHOLD,
        )

        for key in confusion_counts:
            confusion_counts[key] += counts[key]

        running_total_loss += (
            total_loss.item()
        )

        running_bce_loss += (
            bce_loss.item()
        )

        running_dice_loss += (
            dice_loss.item()
        )

        batch_count += 1

    metrics = calculate_metrics(
        confusion_counts
    )

    metrics.update(
        {
            "loss": (
                running_total_loss
                / max(batch_count, 1)
            ),
            "bce_loss": (
                running_bce_loss
                / max(batch_count, 1)
            ),
            "dice_loss": (
                running_dice_loss
                / max(batch_count, 1)
            ),
        }
    )

    return metrics


# =========================================================
# Training
# =========================================================
def main():
    set_random_seeds(RANDOM_SEED)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        use_mixed_precision = False
    else:
        device = torch.device("cuda")
        use_mixed_precision = True

    print("\n" + "=" * 72)
    print("CLIMATEADAPT COMPACT U-NET TRAINING")
    print("=" * 72)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        total_vram = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{total_vram:.2f} GB"
        )

    print(
        "Mixed precision:",
        use_mixed_precision,
    )

    # -----------------------------------------------------
    # Load tile metadata and manifest
    # -----------------------------------------------------
    metadata_path = (
        TILE_FOLDER / "tile_metadata.json"
    )

    manifest_path = (
        TILE_FOLDER / "tile_manifest.csv"
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        tile_metadata = json.load(file)

    manifest = pd.read_csv(
        manifest_path
    )

    input_channels = int(
        tile_metadata["number_of_channels"]
    )

    training_manifest = manifest[
        manifest["split"] == "train"
    ].copy()

    validation_manifest = manifest[
        manifest["split"] == "validation"
    ].copy()

    if training_manifest.empty:
        raise ValueError(
            "Training manifest is empty."
        )

    if validation_manifest.empty:
        raise ValueError(
            "Validation manifest is empty."
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
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    print(
        f"Training tiles: "
        f"{len(training_dataset)}"
    )

    print(
        f"Validation tiles: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Input channels: "
        f"{input_channels}"
    )

    # -----------------------------------------------------
    # Calculate class weighting
    # -----------------------------------------------------
    positive_weight_value = (
        calculate_positive_weight(
            training_dataset
        )
    )

    positive_weight = torch.tensor(
        positive_weight_value,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 1)

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------
    model = CompactUNet(
        input_channels=input_channels,
        base_channels=BASE_CHANNELS,
        dropout_probability=(
            DROPOUT_PROBABILITY
        ),
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=0.000001,
        )
    )

    gradient_scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=use_mixed_precision,
        )
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    best_validation_dice = -1.0
    epochs_without_improvement = 0
    history_records = []

    best_model_path = (
        MODEL_FOLDER
        / "climateadapt_unet_best.pt"
    )

    start_time = time.time()

    for epoch in range(
        1,
        MAXIMUM_EPOCHS + 1,
    ):
        epoch_start = time.time()

        training_metrics = run_epoch(
            model=model,
            data_loader=training_loader,
            device=device,
            positive_weight=positive_weight,
            optimizer=optimizer,
            gradient_scaler=(
                gradient_scaler
            ),
            use_mixed_precision=(
                use_mixed_precision
            ),
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            device=device,
            positive_weight=positive_weight,
            optimizer=None,
            gradient_scaler=None,
            use_mixed_precision=(
                use_mixed_precision
            ),
        )

        scheduler.step(
            validation_metrics["dice"]
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = (
            time.time() - epoch_start
        )

        history_record = {
            "epoch": epoch,
            "learning_rate": (
                current_learning_rate
            ),
            "train_loss": (
                training_metrics["loss"]
            ),
            "train_dice": (
                training_metrics["dice"]
            ),
            "train_precision": (
                training_metrics["precision"]
            ),
            "train_recall": (
                training_metrics["recall"]
            ),
            "train_iou": (
                training_metrics["iou"]
            ),
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_dice": (
                validation_metrics["dice"]
            ),
            "validation_precision": (
                validation_metrics[
                    "precision"
                ]
            ),
            "validation_recall": (
                validation_metrics["recall"]
            ),
            "validation_iou": (
                validation_metrics["iou"]
            ),
            "epoch_seconds": epoch_seconds,
        }

        history_records.append(
            history_record
        )

        print(
            f"Epoch {epoch:02d}/{MAXIMUM_EPOCHS} | "
            f"train loss={training_metrics['loss']:.4f} "
            f"dice={training_metrics['dice']:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} "
            f"dice={validation_metrics['dice']:.4f} "
            f"precision={validation_metrics['precision']:.4f} "
            f"recall={validation_metrics['recall']:.4f} | "
            f"lr={current_learning_rate:.6f} | "
            f"{epoch_seconds:.1f}s"
        )

        if (
            validation_metrics["dice"]
            > best_validation_dice + 0.0001
        ):
            best_validation_dice = (
                validation_metrics["dice"]
            )

            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": (
                    model.state_dict()
                ),
                "input_channels": (
                    input_channels
                ),
                "base_channels": (
                    BASE_CHANNELS
                ),
                "dropout_probability": (
                    DROPOUT_PROBABILITY
                ),
                "epoch": epoch,
                "validation_metrics": (
                    validation_metrics
                ),
                "positive_weight": (
                    positive_weight_value
                ),
                "decision_threshold": (
                    DECISION_THRESHOLD
                ),
                "tile_metadata": (
                    tile_metadata
                ),
            }

            torch.save(
                checkpoint,
                best_model_path,
            )

            print(
                "  Saved new best checkpoint."
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping: "
                "validation Dice did not improve "
                f"for {EARLY_STOPPING_PATIENCE} "
                "epochs."
            )
            break

    total_training_seconds = (
        time.time() - start_time
    )

    # -----------------------------------------------------
    # Save final model and history
    # -----------------------------------------------------
    final_model_path = (
        MODEL_FOLDER
        / "climateadapt_unet_final.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "input_channels": (
                input_channels
            ),
            "base_channels": (
                BASE_CHANNELS
            ),
            "dropout_probability": (
                DROPOUT_PROBABILITY
            ),
            "positive_weight": (
                positive_weight_value
            ),
            "tile_metadata": (
                tile_metadata
            ),
        },
        final_model_path,
    )

    history = pd.DataFrame(
        history_records
    )

    history_path = (
        OUTPUT_FOLDER
        / "training_history.csv"
    )

    history.to_csv(
        history_path,
        index=False,
    )

    # -----------------------------------------------------
    # Plot learning curves
    # -----------------------------------------------------
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

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

    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        history["epoch"],
        history["train_dice"],
        label="Training",
    )

    axes[0, 1].plot(
        history["epoch"],
        history["validation_dice"],
        label="Validation",
    )

    axes[0, 1].set_title("Dice score")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(
        history["epoch"],
        history["validation_precision"],
        label="Validation precision",
        color="#2ca02c",
    )

    axes[1, 0].plot(
        history["epoch"],
        history["validation_recall"],
        label="Validation recall",
        color="#d62728",
    )

    axes[1, 0].set_title(
        "Validation precision and recall"
    )

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(
        history["epoch"],
        history["train_iou"],
        label="Training IoU",
    )

    axes[1, 1].plot(
        history["epoch"],
        history["validation_iou"],
        label="Validation IoU",
    )

    axes[1, 1].set_title(
        "Intersection over Union"
    )

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    figure.suptitle(
        "ClimateAdapt AI – Compact U-Net Training",
        fontsize=16,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    curve_path = (
        EVIDENCE_FOLDER
        / "unet_training_curves.png"
    )

    plt.savefig(
        curve_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print("\n" + "=" * 72)
    print("TRAINING COMPLETED")
    print("=" * 72)
    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.4f}"
    )
    print(
        f"Training time: "
        f"{total_training_seconds / 60:.2f} minutes"
    )
    print(
        f"Best model: "
        f"{best_model_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Final model: "
        f"{final_model_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"History: "
        f"{history_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Training curves: "
        f"{curve_path.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 72)
    print(
        "The independent test dataset was not used "
        "during training."
    )


if __name__ == "__main__":
    main()