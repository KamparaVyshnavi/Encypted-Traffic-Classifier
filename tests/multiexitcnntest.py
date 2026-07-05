"""
===============================================================================
Multi-Exit CNN Evaluation
===============================================================================

Part 1

✓ Load Dataset
✓ Load Model
✓ Early Exit Trace
✓ Actual Exit Verification

Future Parts

Part 2
    Full Dataset Evaluation

Part 3
    Latency Benchmark

Part 4
    Final Summary
"""

from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from model.dataset import TrafficDataset
from model.inference import MultiExitInference

from utils.config import (
    DATASET_ROOT,
    MULTI_EXIT_CHECKPOINT_DIR,
    MULTI_EXIT_BEST_MODEL_NAME,
    BATCH_SIZE,
)


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

THRESHOLD = 0.90

TRACE_SAMPLES = 20


# --------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------

def separator():

    print("=" * 80)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():

    separator()

    print("MULTI-EXIT CNN EVALUATION")

    separator()

    print()

    print("Loading Dataset...")

    dataset = TrafficDataset(

        dataset_root=DATASET_ROOT,

        verbose=False,

    )

    dataloader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

    )

    print(f"Samples : {len(dataset)}")

    print()

    checkpoint = (

        Path(MULTI_EXIT_CHECKPOINT_DIR)

        / MULTI_EXIT_BEST_MODEL_NAME

    )

    print("Loading Model...")

    inference = MultiExitInference(

        checkpoint_path=checkpoint,

    )

    print("Model Loaded")

    print()

    separator()

    print("EARLY EXIT TRACE")

    separator()

    print()

    print(

        f"Threshold : {THRESHOLD:.2f}"

    )

    print()

    import random

    indices = random.sample(
        range(len(dataset)),
        TRACE_SAMPLES,
    )

    for i in indices:

        sequence, label = dataset[i]

        start = time.perf_counter()

        result = inference.predict(

            sequence,

            threshold=THRESHOLD,

        )

        end = time.perf_counter()

        print("-" * 60)

        print(f"Sample {i+1}")

        print()

        print(

            f"Ground Truth : "

            f"{label.item()}"

        )

        print(

            f"Prediction   : "

            f"{result['prediction']}"

        )

        print(

            f"Confidence   : "

            f"{result['confidence']:.4f}"

        )

        print(

            f"Exit Used    : "

            f"{result['exit']}"

        )

        print(

            f"Blocks Used  : "

            f"{result['blocks_executed']} / 4"

        )

        print(

            f"Inference    : "

            f"{(end-start)*1000:.3f} ms"

        )

        if result["prediction"] == label.item():

            print("Result       : CORRECT")

        else:

            print("Result       : WRONG")

    print()

        # ----------------------------------------------------------------
    # Full Dataset Evaluation
    # ----------------------------------------------------------------

    separator()

    print("FULL DATASET EVALUATION")

    separator()

    total_samples = 0

    correct_predictions = 0

    exit_counts = {

        "exit1": 0,

        "exit2": 0,

        "exit3": 0,

        "final": 0,
    }

    total_blocks = 0

    total_latency = 0.0

    for sequences, labels in dataloader:

        for sequence, label in zip(sequences, labels):

            start = time.perf_counter()

            result = inference.predict(

                sequence,

                threshold=THRESHOLD,

            )

            end = time.perf_counter()

            latency = (end - start) * 1000

            total_latency += latency

            total_samples += 1

            if result["prediction"] == label.item():

                correct_predictions += 1

            exit_counts[result["exit"]] += 1

            total_blocks += result["blocks_executed"]

    # ------------------------------------------------------------

    accuracy = (

        correct_predictions /

        total_samples

    )

    average_blocks = (

        total_blocks /

        total_samples

    )

    average_packets = (

        average_blocks *

        5
    )

    computation_reduction = (

        20 - average_packets

    ) / 20

    average_latency = (

        total_latency /

        total_samples

    )

    # ------------------------------------------------------------

    print()

    print(f"Dataset Size              : {total_samples}")

    print(f"Threshold                : {THRESHOLD:.2f}")

    print()

    print(f"Accuracy                 : {accuracy:.4%}")

    print(f"Average Latency          : {average_latency:.3f} ms")

    print(f"Average Blocks Executed  : {average_blocks:.2f} / 4")

    print(f"Average Packets Used     : {average_packets:.2f} / 20")

    print(

        f"Computation Reduction    : "

        f"{computation_reduction:.2%}"

    )

    print()

    print("Exit Distribution")

    print("-" * 40)

    for exit_name, count in exit_counts.items():

        percentage = (

            count /

            total_samples

        ) * 100

        print(

            f"{exit_name:<8}"

            f": "

            f"{count:5d}"

            f" ({percentage:.2f}%)"

        )

    print()

    separator()

    print("FULL DATASET EVALUATION COMPLETED")

    separator()
    # ==============================================================
    # LATENCY BENCHMARK
    # ==============================================================

    separator()

    print("LATENCY BENCHMARK")

    separator()

    full_latency = 0.0

    early_latency = 0.0

    NUM_RUNS = 5

    for sequence, _ in dataset:

        sequence = sequence.unsqueeze(0)

        sequence = sequence.to(inference.device)

        # ------------------------------------------------------
        # Full Forward Pass
        # ------------------------------------------------------

        start = time.perf_counter()

        for _ in range(NUM_RUNS):

            _ = inference.model.forward(sequence)

        end = time.perf_counter()

        full_latency += (

            (end - start)

            / NUM_RUNS

        )

        # ------------------------------------------------------
        # Early Exit Forward
        # ------------------------------------------------------

        start = time.perf_counter()

        for _ in range(NUM_RUNS):

            _ = inference.model.early_exit_forward(

                sequence,

                threshold=THRESHOLD,

            )

        end = time.perf_counter()

        early_latency += (

            (end - start)

            / NUM_RUNS

        )

    full_latency /= len(dataset)

    early_latency /= len(dataset)

    latency_reduction = (

        full_latency -

        early_latency

    ) / full_latency

    print()

    print(

        f"Average Full Inference     : "

        f"{full_latency*1000:.4f} ms"

    )

    print(

        f"Average Early Exit         : "

        f"{early_latency*1000:.4f} ms"

    )

    print()

    print(

        f"Latency Reduction          : "

        f"{latency_reduction:.2%}"

    )

    separator()

    print("LATENCY BENCHMARK COMPLETED")

    separator()


if __name__ == "__main__":

    main()