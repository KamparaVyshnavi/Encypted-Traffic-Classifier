from evaluation.model_evaluator import evaluate_model

from utils.config import (
    MULTI_EXIT_CHECKPOINT_DIR,
    MULTI_EXIT_BEST_MODEL_NAME,
    NOVELTY_TEST_DATASET,
)
from pathlib import Path

checkpoint = (
    Path(MULTI_EXIT_CHECKPOINT_DIR)
    / MULTI_EXIT_BEST_MODEL_NAME
)

def print_experiment_info():

    print("=" * 75)
    print("MULTI-EXIT CROSS NETWORK EVALUATION")
    print("=" * 75)

    print()

    print("Checkpoint :", checkpoint)
    print("Dataset    :", NOVELTY_TEST_DATASET)

    print()


def run_cross_network_test():

    print_experiment_info()

    result = evaluate_model(

        checkpoint_path=checkpoint,

        dataset_root=NOVELTY_TEST_DATASET,

        multi_exit=True,

        threshold=0.90,

        print_summary=True,
    )

    return result


if __name__ == "__main__":

    run_cross_network_test()