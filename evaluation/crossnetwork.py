"""
===============================================================================
Cross Network Evaluation
===============================================================================

Evaluates the cross-network robustness of the proposed
Handshake-Based Temporal Normalization.

Experiment

Train:
    ISCX

Test:
    VTAN

This script NEVER trains a model.

It simply loads already-trained models and evaluates them
on unseen network traffic.

Uses:
    model_evaluator.py
"""

from pathlib import Path

from evaluation.model_evaluator import (
    evaluate_model,
)

from utils.config import (
    BASELINE_MODEL_PATH,
    BASELINE_TEST_DATASET,
    NOVELTY_MODEL_PATH,
    NOVELTY_TEST_DATASET,
)

# =============================================================================
# Utility
# =============================================================================

def print_comparison(
    baseline_result,
    novelty_result,
):

    print()

    print("=" * 75)
    print("CROSS NETWORK COMPARISON")
    print("=" * 75)

    print()

    print(
        f"{'Metric':<20}"
        f"{'Baseline':<15}"
        f"{'Novelty-1':<15}"
    )

    print("-" * 75)

    print(
        f"{'Accuracy':<20}"
        f"{baseline_result.accuracy:.4%}"
        f"{'':<6}"
        f"{novelty_result.accuracy:.4%}"
    )

    print(
        f"{'Precision':<20}"
        f"{baseline_result.precision:.4%}"
        f"{'':<6}"
        f"{novelty_result.precision:.4%}"
    )

    print(
        f"{'Recall':<20}"
        f"{baseline_result.recall:.4%}"
        f"{'':<6}"
        f"{novelty_result.recall:.4%}"
    )

    print(
        f"{'F1 Score':<20}"
        f"{baseline_result.f1_score:.4%}"
        f"{'':<6}"
        f"{novelty_result.f1_score:.4%}"
    )

    print()

    improvement = (
        novelty_result.accuracy
        -
        baseline_result.accuracy
    )

    print(
        f"Accuracy Improvement : "
        f"{improvement:.4%}"
    )

    print()

    print("=" * 75)

# =============================================================================
# Experiment Information
# =============================================================================

def print_experiment_info(
    title,
    checkpoint,
    dataset,
):

    print()

    print("=" * 75)
    print(title)
    print("=" * 75)

    print()

    print(
        "Checkpoint :",
        checkpoint,
    )

    print(
        "Dataset    :",
        dataset,
    )

    print()
# =============================================================================
# Cross Network Evaluation
# =============================================================================

def run_cross_network_test():

    print()

    print_experiment_info(

    "BASELINE MODEL",

    BASELINE_MODEL_PATH,

    BASELINE_TEST_DATASET,
)

    baseline_result = evaluate_model(

    checkpoint_path=BASELINE_MODEL_PATH,

    dataset_root=BASELINE_TEST_DATASET,

    print_summary=True,
)

    print()

    print_experiment_info(

    "NOVELTY-1 MODEL",

    NOVELTY_MODEL_PATH,

    NOVELTY_TEST_DATASET,
)
    novelty_result = evaluate_model(

    checkpoint_path=NOVELTY_MODEL_PATH,

    dataset_root=NOVELTY_TEST_DATASET,

    print_summary=True,
)

    print_comparison(

        baseline_result,

        novelty_result,
    )

    return (

        baseline_result,

        novelty_result,
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    run_cross_network_test()