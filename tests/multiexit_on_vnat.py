"""
===============================================================================
Cross-Network Threshold Sweep
===============================================================================

Evaluates the trained Multi-Exit CNN on the VN80 normalized dataset
using different confidence thresholds.

Train
-----
ISCX Normalized

Test
----
VN80 Normalized

Purpose
-------
Select the best operating threshold for unseen networks.
"""

from pathlib import Path

from evaluation.model_evaluator import evaluate_model

from utils.config import (
    MULTI_EXIT_CHECKPOINT_DIR,
    MULTI_EXIT_BEST_MODEL_NAME,
    NOVELTY_TEST_DATASET,
)

THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]

checkpoint = (
    Path(MULTI_EXIT_CHECKPOINT_DIR)
    / MULTI_EXIT_BEST_MODEL_NAME
)


def main():

    print("=" * 100)
    print("MULTI-EXIT CROSS-NETWORK THRESHOLD SWEEP")
    print("=" * 100)
    print()

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Exit1':<10}"
        f"{'Exit2':<10}"
        f"{'Exit3':<10}"
        f"{'Final':<10}"
        f"{'AvgBlocks':<12}"
    )

    print("-" * 100)

    for threshold in THRESHOLDS:

        result = evaluate_model(

            checkpoint_path=checkpoint,

            dataset_root=NOVELTY_TEST_DATASET,

            multi_exit=True,

            threshold=threshold,

            print_summary=False,
        )

        evaluator = result.evaluator

        total = sum(
            evaluator.exit_counts.values()
        )

        e1 = evaluator.exit_counts["exit1"] / total
        e2 = evaluator.exit_counts["exit2"] / total
        e3 = evaluator.exit_counts["exit3"] / total
        ef = evaluator.exit_counts["final"] / total

        avg_blocks = (
            evaluator.total_blocks / total
        )

        print(
            f"{threshold:<12.2f}"
            f"{result.accuracy:<12.4%}"
            f"{e1:<10.2%}"
            f"{e2:<10.2%}"
            f"{e3:<10.2%}"
            f"{ef:<10.2%}"
            f"{avg_blocks:<12.2f}"
        )

    print()
    print("=" * 100)


if __name__ == "__main__":

    main()