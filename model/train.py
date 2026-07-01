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
from model.temporal_cnn import TemporalCNN

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
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    LATEST_MODEL_NAME,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=PIN_MEMORY,
        num_workers=0,
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
    Creates the Temporal CNN.
    """

    model = TemporalCNN()

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
    """
    Trains the model for one epoch.

    Returns
    -------
    epoch_loss
    epoch_accuracy
    """

    model.train()

    running_loss = 0.0

    correct_predictions = 0

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

        logits = model(sequences)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = (
        correct_predictions / total_samples
    )

    return epoch_loss, epoch_accuracy

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
):
    """
    Evaluates the model.

    Returns
    -------
    validation_loss
    validation_accuracy
    """

    model.eval()

    running_loss = 0.0

    correct_predictions = 0

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

        logits = model(sequences)

        loss = criterion(
            logits,
            labels,
        )

        running_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    validation_loss = (
        running_loss / len(dataloader)
    )

    validation_accuracy = (
        correct_predictions / total_samples
    )

    return validation_loss, validation_accuracy

def print_epoch_summary(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    validation_loss: float,
    validation_accuracy: float,
):
    """
    Prints epoch statistics.
    """

    print("-" * 70)

    print(f"Epoch {epoch}")

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : {train_accuracy:.4%}"
    )

    print(
        f"Validation Loss : {validation_loss:.4f}"
    )

    print(
        f"Validation Acc. : {validation_accuracy:.4%}"
    )

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

    checkpoint_directory = Path(CHECKPOINT_DIR)

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
    "train_accuracy": [],
    "validation_accuracy": [],
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

        history["train_accuracy"].append(train_accuracy)

        history["validation_accuracy"].append(validation_accuracy)

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
            checkpoint_directory / LATEST_MODEL_NAME
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            validation_accuracy,
            latest_checkpoint,
        )

        # ------------------------------------------------------------
        # Save Best Model
        # ------------------------------------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            best_checkpoint = (
                checkpoint_directory / BEST_MODEL_NAME
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                validation_accuracy,
                best_checkpoint,
            )

            print(
                f"New Best Model "
                f"({validation_accuracy:.4%})"
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

    history_path = checkpoint_directory / "training_history.json"

    with open(history_path, "w") as fp:
        json.dump(history, fp, indent=4)
    return history

# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    train()