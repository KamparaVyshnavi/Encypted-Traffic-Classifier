"""
===============================================================================
Ablation Study
===============================================================================

Compares the contribution of each component of the proposed framework.

Experiments
-----------
1. Baseline CNN
2. Novelty-1 (Temporal Normalization)
3. Novelty-1 + Novelty-2 (Multi-Exit CNN)

Outputs
-------
Accuracy
Precision
Recall
F1 Score

For Multi-Exit:
    Exit Distribution
    Average Blocks
"""

from pathlib import Path

from evaluation.model_evaluator import evaluate_model

from utils.config import (
    BASELINE_MODEL_PATH,
    BASELINE_TEST_DATASET,
    NOVELTY_MODEL_PATH,
    NOVELTY_TEST_DATASET,
    MULTI_EXIT_CHECKPOINT_DIR,
    MULTI_EXIT_BEST_MODEL_NAME,
)

# ------------------------------------------------------------
# Multi-Exit Checkpoint
# ------------------------------------------------------------

MULTI_EXIT_MODEL_PATH = (
    Path(MULTI_EXIT_CHECKPOINT_DIR)
    / MULTI_EXIT_BEST_MODEL_NAME
)

THRESHOLD = 0.70


# ------------------------------------------------------------
# Pretty Printing
# ------------------------------------------------------------

def separator():

    print("=" * 110)


def print_header():

    separator()

    print("ABLATION STUDY")

    separator()

    print()


# ------------------------------------------------------------
# Individual Experiments
# ------------------------------------------------------------

def run_baseline():

    print()

    separator()

    print("Experiment 1 : Baseline CNN")

    separator()

    return evaluate_model(

        checkpoint_path=BASELINE_MODEL_PATH,

        dataset_root=BASELINE_TEST_DATASET,

        print_summary=False,
    )


def run_novelty1():

    print()

    separator()

    print("Experiment 2 : Temporal Normalization")

    separator()

    return evaluate_model(

        checkpoint_path=NOVELTY_MODEL_PATH,

        dataset_root=NOVELTY_TEST_DATASET,

        print_summary=False,
    )


def run_multiexit():

    print()

    separator()

    print("Experiment 3 : Multi-Exit CNN")

    separator()

    return evaluate_model(

        checkpoint_path=MULTI_EXIT_MODEL_PATH,

        dataset_root=NOVELTY_TEST_DATASET,

        multi_exit=True,

        threshold=THRESHOLD,

        print_summary=False,
    )
# ------------------------------------------------------------
# Comparison Table
# ------------------------------------------------------------

def print_results_table(
    baseline,
    novelty1,
    multiexit,
):

    separator()

    print("ABLATION RESULTS")

    separator()

    print()

    print(
        f"{'Metric':<18}"
        f"{'Baseline':<15}"
        f"{'Novelty-1':<15}"
        f"{'Novelty-1 + 2':<18}"
    )

    print("-" * 70)

    print(
        f"{'Accuracy':<18}"
        f"{baseline.accuracy:<15.4%}"
        f"{novelty1.accuracy:<15.4%}"
        f"{multiexit.accuracy:<18.4%}"
    )

    print(
        f"{'Precision':<18}"
        f"{baseline.precision:<15.4%}"
        f"{novelty1.precision:<15.4%}"
        f"{multiexit.precision:<18.4%}"
    )

    print(
        f"{'Recall':<18}"
        f"{baseline.recall:<15.4%}"
        f"{novelty1.recall:<15.4%}"
        f"{multiexit.recall:<18.4%}"
    )

    print(
        f"{'F1 Score':<18}"
        f"{baseline.f1_score:<15.4%}"
        f"{novelty1.f1_score:<15.4%}"
        f"{multiexit.f1_score:<18.4%}"
    )

    print()

    separator()

    print("IMPROVEMENTS")

    separator()

    print()

    novelty1_gain = (

        novelty1.accuracy

        - baseline.accuracy

    )

    novelty2_gain = (

        multiexit.accuracy

        - novelty1.accuracy

    )

    overall_gain = (

        multiexit.accuracy

        - baseline.accuracy

    )

    print(

        f"Novelty-1 Improvement : "

        f"{novelty1_gain:.4%}"

    )

    print(

        f"Novelty-2 Improvement : "

        f"{novelty2_gain:.4%}"

    )

    print(

        f"Overall Improvement   : "

        f"{overall_gain:.4%}"

    )

    print()

    separator()

    print("MULTI-EXIT STATISTICS")

    separator()

    evaluator = multiexit.evaluator

    total = sum(

        evaluator.exit_counts.values()

    )

    for exit_name, count in evaluator.exit_counts.items():

        print(

            f"{exit_name:<8}"

            f": "

            f"{count:5d}"

            f" ({count/total:.2%})"

        )

    print()

    average_blocks = (

        evaluator.total_blocks

        / total

    )

    average_packets = (

        average_blocks

        * 5

    )

    computation_reduction = (

        20 - average_packets

    ) / 20

    print(

        f"Average Blocks Executed : "

        f"{average_blocks:.2f}"

    )

    print(

        f"Average Packets Used    : "

        f"{average_packets:.2f}"

    )

    print(

        f"Computation Reduction   : "

        f"{computation_reduction:.2%}"

    )

    print()

    separator()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print_header()

    baseline = run_baseline()

    novelty1 = run_novelty1()

    multiexit = run_multiexit()

    print_results_table(

        baseline,

        novelty1,

        multiexit,

    )


if __name__ == "__main__":

    main()