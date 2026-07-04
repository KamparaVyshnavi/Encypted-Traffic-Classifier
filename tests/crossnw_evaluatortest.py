"""
=============================================================
Model Evaluator Test
=============================================================

Verifies that the evaluation pipeline works correctly.

Checks
------
1. Checkpoint loading
2. Dataset loading
3. DataLoader creation
4. Forward pass
5. Prediction generation
6. Metric computation
"""

from pathlib import Path

import torch

from evaluation.model_evaluator import ModelEvaluator

from utils.config import (
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    DATASET_ROOT,
)


def main():

    print("=" * 70)
    print("MODEL EVALUATOR TEST")
    print("=" * 70)

    checkpoint = (
        Path(CHECKPOINT_DIR)
        / BEST_MODEL_NAME
    )

    print()
    print("Checkpoint")
    print("----------")

    if checkpoint.exists():

        print("✓ Found :", checkpoint)

    else:

        raise FileNotFoundError(checkpoint)

    evaluator = ModelEvaluator()

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------

    print()
    print("Loading model...")

    evaluator.load_model(checkpoint)

    print("✓ Model loaded")

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------

    print()
    print("Loading dataset...")

    evaluator.load_dataset(DATASET_ROOT)

    print("✓ Dataset loaded")

    print()

    print(
        "Samples :",
        len(evaluator.dataset)
    )

    print(
        "Classes :",
        evaluator.dataset.num_classes()
    )

    # ----------------------------------------------------------
    # First batch
    # ----------------------------------------------------------

    print()
    print("Testing first batch...")

    loader = evaluator.dataloader

    sequences, labels = next(iter(loader))

    print(
        "Input Shape :",
        tuple(sequences.shape)
    )

    print(
        "Label Shape :",
        tuple(labels.shape)
    )
    from utils.config import DEVICE
    device = next(
    evaluator.model.parameters()
).device

    sequences = sequences.to(device)

    with torch.no_grad():

        logits = evaluator.model(
            sequences
        )

    print()

    print(
        "Output Shape :",
        tuple(logits.shape)
    )

    assert logits.shape[0] == sequences.shape[0]

    assert logits.shape[1] == evaluator.dataset.num_classes()

    print("✓ Forward pass successful")

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    print()
    print("Generating predictions...")

    predictions, targets = evaluator.predict()

    print(
        "Predictions :",
        len(predictions)
    )

    print(
        "Targets     :",
        len(targets)
    )

    assert len(predictions) == len(targets)

    print("✓ Prediction successful")

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    print()
    print("Computing metrics...")

    result = evaluator.compute_metrics(
        predictions,
        targets,
    )

    print()

    print(
        f"Accuracy : {result.accuracy:.4%}"
    )

    print(
        f"Precision : {result.precision:.4%}"
    )

    print(
        f"Recall : {result.recall:.4%}"
    )

    print(
        f"F1 Score : {result.f1_score:.4%}"
    )

    print()

    print(
        "Confusion Matrix Shape :",
        result.confusion_matrix.shape
    )

    print()

    print("=" * 70)
    print("MODEL EVALUATOR TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    main()