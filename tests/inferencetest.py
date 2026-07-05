"""
===============================================================================
Multi-Exit Inference Engine Test
===============================================================================

Verifies

✓ Model loading
✓ Checkpoint loading
✓ Single prediction
✓ Batch prediction
✓ Dataset evaluation
✓ Statistics generation
✓ Threshold sweep
✓ Numerical validity

Run

python -m tests.test_inference
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.dataset import TrafficDataset
from model.inference import MultiExitInference

from utils.config import (
    DATASET_ROOT,
    BATCH_SIZE,
    MULTI_EXIT_CHECKPOINT_DIR,
    MULTI_EXIT_BEST_MODEL_NAME,
)


def separator():
    print("=" * 80)


def check(condition, message):

    if condition:
        print(f"✓ {message}")
    else:
        raise AssertionError(f"✗ {message}")


def main():

    separator()
    print("MULTI-EXIT INFERENCE ENGINE TEST")
    separator()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset = TrafficDataset(
        dataset_root=DATASET_ROOT,
        verbose=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    check(len(dataset) > 0, "Dataset Loaded")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    checkpoint = (
        Path(MULTI_EXIT_CHECKPOINT_DIR)
        / MULTI_EXIT_BEST_MODEL_NAME
    )

    check(
        checkpoint.exists(),
        "Checkpoint Found",
    )

    # ------------------------------------------------------------------
    # Inference Engine
    # ------------------------------------------------------------------

    inference = MultiExitInference(
        checkpoint_path=checkpoint,
    )

    check(True, "Inference Engine Created")

    # ------------------------------------------------------------------
    # Single Prediction
    # ------------------------------------------------------------------

    sequence, label = dataset[0]

    result = inference.predict(
        sequence,
        threshold=0.70,
    )

    check(
        isinstance(result, dict),
        "Single Prediction Returns Dictionary",
    )

    check(
        "prediction" in result,
        "Prediction Exists",
    )

    check(
        "confidence" in result,
        "Confidence Exists",
    )

    check(
        "exit" in result,
        "Exit Exists",
    )

    print()

    print("Single Prediction")

    print("Ground Truth     :", label.item())

    print("Prediction       :", result["prediction"])

    print("Confidence       :", f"{result['confidence']:.4f}")

    print("Exit             :", result["exit"])

    print("Blocks Executed  :", result["blocks_executed"])

    # ------------------------------------------------------------------
    # Batch Prediction
    # ------------------------------------------------------------------

    sequences = []

    labels = []

    for i in range(8):

        s, l = dataset[i]

        sequences.append(s)

        labels.append(l)

    sequences = torch.stack(sequences)

    results = inference.predict_batch(
        sequences,
        threshold=0.90,
    )
    for i in range(100):

        result = inference.predict(
            dataset[i][0],
            threshold=0.70,
        )

        if result["exit"] != "final":

            print(
                i,
                result["exit"],
                result["confidence"],
                result["blocks_executed"],
            )

            break

    check(
        len(results) == 8,
        "Batch Prediction Size Correct",
    )

    print()

    print("Batch Prediction")

    for i, r in enumerate(results):

        print(
    f"{i+1:2d}. "
    f"Pred={r['prediction']} "
    f"Conf={r['confidence']:.3f} "
    f"Exit={r['exit']} "
    f"Blocks={r['blocks_executed']}"
)

    # ------------------------------------------------------------------
    # Dataset Evaluation
    # ------------------------------------------------------------------

    print()

    print("Running Dataset Evaluation...")

    statistics = inference.evaluate(
        dataloader,
        threshold=0.90,
    )

    check(
        "accuracy" in statistics,
        "Accuracy Computed",
    )

    check(
        "exit_counts" in statistics,
        "Exit Counts Computed",
    )

    check(
        "average_exit" in statistics,
        "Average Exit Computed",
    )

    check(
        "average_packets" in statistics,
        "Average Packets Computed",
    )

    check(
        "latency_reduction" in statistics,
        "Latency Reduction Computed",
    )

    print()

    inference.print_statistics(statistics)

    # ------------------------------------------------------------------
    # Numerical Checks
    # ------------------------------------------------------------------

    total = sum(
        statistics["exit_counts"].values()
    )

    check(
        total == len(dataset),
        "Every Sample Evaluated",
    )

    check(
        0 <= statistics["accuracy"] <= 1,
        "Accuracy Valid",
    )

    check(
        1 <= statistics["average_exit"] <= 4,
        "Average Exit Valid",
    )

    check(
        5 <= statistics["average_packets"] <= 20,
        "Average Packets Valid",
    )

    check(
        0 <= statistics["latency_reduction"] <= 1,
        "Latency Reduction Valid",
    )

    # ------------------------------------------------------------------
    # Threshold Sweep
    # ------------------------------------------------------------------

    print()

    print("Running Threshold Sweep...")

    sweep = inference.sweep_thresholds(
        dataloader,
    )

    check(
        len(sweep) == 7,
        "Threshold Sweep Completed",
    )

    separator()

    print("ALL TESTS PASSED SUCCESSFULLY")

    separator()


if __name__ == "__main__":

    main()