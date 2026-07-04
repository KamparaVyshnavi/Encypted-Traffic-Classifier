"""
===============================================================================
Model Training Pipeline
===============================================================================

Implements the complete training pipeline for the baseline
Temporal CNN.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model.dataset import TrafficDataset
from model.multi_exitcnn import MultiExitCNN
from torch.utils.data import WeightedRandomSampler

from utils.config import (
    DEVICE,
    PIN_MEMORY,
    NON_BLOCKING,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    TRAIN_SPLIT,
    RANDOM_SEED,
    MULTI_EXIT_CHECKPOINT_DIR,
    MULTI_EXIT_BEST_MODEL_NAME,
    MULTI_EXIT_LATEST_MODEL_NAME,
    MULTI_EXIT_HISTORY_NAME,
    DATASET_ROOT,
)

def set_random_seed(seed: int) -> None:
    """
    Sets random seed for reproducible experiments.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dataloaders():
    """
    Creates training and validation dataloaders.
    """

    dataset = TrafficDataset(
    dataset_root=DATASET_ROOT,
    verbose=False,
)

    train_size = int(len(dataset) * TRAIN_SPLIT)

    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(
            RANDOM_SEED
        ),
    )

    # ------------------------------------------------------------
    # Weighted Sampler
    # ------------------------------------------------------------

    train_indices = train_dataset.indices

    train_labels = [
    dataset.label_to_index[
        dataset.labels.iloc[i]["label"]
    ]
    for i in train_indices
]

    from collections import Counter

    class_counts = Counter(train_labels)

    class_weights = {
        label: 1.0 / count
        for label, count in class_counts.items()
    }

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=False,
    pin_memory=PIN_MEMORY,
)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        num_workers=0,
    )

    return train_loader, validation_loader

def build_model():
    """
    Creates the Multi-Exit CNN.
    """

    model = MultiExitCNN()

    model.to(DEVICE)

    return model

def create_optimizer(
    model: nn.Module,
):
    """
    Creates Adam optimizer.
    """

    return Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
def create_loss():
    """
    Creates classification loss.
    """

    return nn.CrossEntropyLoss()
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
):

    model.train()

    running_loss = 0.0

    exit_correct = {
        "exit1": 0,
        "exit2": 0,
        "exit3": 0,
        "final": 0,
    }

    total_samples = 0

    for sequences, labels in dataloader:

        sequences = sequences.to(
            DEVICE,
            non_blocking=NON_BLOCKING,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=NON_BLOCKING,
        )

        optimizer.zero_grad()

        outputs = model(sequences)

        logits = outputs["logits"]

        loss1 = criterion(
            logits["exit1"],
            labels,
        )

        loss2 = criterion(
            logits["exit2"],
            labels,
        )

        loss3 = criterion(
            logits["exit3"],
            labels,
        )

        loss4 = criterion(
            logits["final"],
            labels,
        )

        loss = (
            0.2 * loss1 +
            0.2 * loss2 +
            0.3 * loss3 +
            0.3 * loss4
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += loss.item()

        for exit_name in logits:

            predictions = torch.argmax(
                logits[exit_name],
                dim=1,
            )

            exit_correct[exit_name] += (
                predictions == labels
            ).sum().item()

        total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    exit_accuracy = {

        key: value / total_samples

        for key, value in exit_correct.items()

    }

    return epoch_loss, exit_accuracy
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
):

    model.eval()

    running_loss = 0.0

    exit_correct = {
        "exit1": 0,
        "exit2": 0,
        "exit3": 0,
        "final": 0,
    }

    total_samples = 0

    for sequences, labels in dataloader:

        sequences = sequences.to(
            DEVICE,
            non_blocking=NON_BLOCKING,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=NON_BLOCKING,
        )

        outputs = model(sequences)

        logits = outputs["logits"]

        loss1 = criterion(
            logits["exit1"],
            labels,
        )

        loss2 = criterion(
            logits["exit2"],
            labels,
        )

        loss3 = criterion(
            logits["exit3"],
            labels,
        )

        loss4 = criterion(
            logits["final"],
            labels,
        )

        loss = (
            0.2 * loss1 +
            0.2 * loss2 +
            0.3 * loss3 +
            0.3 * loss4
        )

        running_loss += loss.item()

        for exit_name in logits:

            predictions = torch.argmax(
                logits[exit_name],
                dim=1,
            )

            exit_correct[exit_name] += (
                predictions == labels
            ).sum().item()

        total_samples += labels.size(0)

    validation_loss = running_loss / len(dataloader)

    exit_accuracy = {

        key: value / total_samples

        for key, value in exit_correct.items()

    }

    return validation_loss, exit_accuracy

def print_epoch_summary(
    epoch,
    train_loss,
    train_accuracy,
    validation_loss,
    validation_accuracy,
):

    print("-" * 70)

    print(f"Epoch {epoch}")

    print(f"Train Loss      : {train_loss:.4f}")

    print()

    print("Training Accuracy")

    for key, value in train_accuracy.items():

        print(f"  {key:<6}: {value:.4%}")

    print()

    print(f"Validation Loss : {validation_loss:.4f}")

    print()

    print("Validation Accuracy")

    for key, value in validation_accuracy.items():

        print(f"  {key:<6}: {value:.4%}")

    print("-" * 70)
# ---------------------------------------------------------------------
# Checkpoint Utilities
# ---------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_accuracy: float,
    filename: Path,
) -> None:
    """
    Saves model checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_accuracy": validation_accuracy,
    }

    torch.save(checkpoint, filename)

# ---------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------

def train():

    set_random_seed(RANDOM_SEED)

    checkpoint_directory = Path(MULTI_EXIT_CHECKPOINT_DIR)

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_loader, validation_loader = create_dataloaders()

    model = build_model()

    optimizer = create_optimizer(model)

    criterion = create_loss()

    best_validation_accuracy = float("-inf")

    history = {
    "train_loss": [],
    "validation_loss": [],
    "train_accuracy": {
        "exit1": [],
        "exit2": [],
        "exit3": [],
        "final": [],
    },
    "validation_accuracy": {
        "exit1": [],
        "exit2": [],
        "exit3": [],
        "final": [],
    },
}

    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    print(f"Device              : {DEVICE}")
    print(f"Training Samples    : {len(train_loader.dataset)}")
    print(f"Validation Samples  : {len(validation_loader.dataset)}")
    print(f"Epochs              : {NUM_EPOCHS}")
    print(f"Batch Size          : {BATCH_SIZE}")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
        )

        history["train_loss"].append(train_loss)

        history["validation_loss"].append(validation_loss)

        for key in train_accuracy:

            history["train_accuracy"][key].append(
                train_accuracy[key]
            )

            history["validation_accuracy"][key].append(
                validation_accuracy[key]
            )

        print_epoch_summary(
            epoch,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )

        # ------------------------------------------------------------
        # Save Latest Model
        # ------------------------------------------------------------

        latest_checkpoint = (
            checkpoint_directory / MULTI_EXIT_LATEST_MODEL_NAME
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            validation_accuracy["final"],
            latest_checkpoint,
        )

        # ------------------------------------------------------------
        # Save Best Model
        # ------------------------------------------------------------

        if validation_accuracy["final"] > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy["final"]

            best_checkpoint = (
                checkpoint_directory / MULTI_EXIT_BEST_MODEL_NAME
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                validation_accuracy["final"],
                best_checkpoint,
            )

            print(
    f"New Best Model "
    f"({validation_accuracy['final']:.4%})"
)

    print()

    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best Validation Accuracy : "
        f"{best_validation_accuracy:.4%}"
    )
    import json

    history_path = checkpoint_directory / MULTI_EXIT_HISTORY_NAME

    with open(history_path, "w") as fp:
        json.dump(history, fp, indent=4)
    return history

# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    train()