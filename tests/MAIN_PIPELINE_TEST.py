"""
===============================================================================
Offline System Demonstration
===============================================================================

Runs the complete encrypted traffic classification system
on the processed ISCX dataset.

Unlike main.py, this script knows the ground truth labels,
allowing accuracy and per-class statistics to be displayed.

Pipeline

Dataset
    ↓
Multi-Exit CNN
    ↓
Early Exit
    ↓
Prediction
    ↓
Statistics
"""

from pathlib import Path
from collections import defaultdict
import time

import torch
from torch.utils.data import DataLoader

from model.dataset import TrafficDataset
from model.inference import MultiExitInference

from utils.config import (
    BATCH_SIZE,
    MULTI_EXIT_CHECKPOINT_DIR,
    MULTI_EXIT_BEST_MODEL_NAME,
    DATASET_ROOT,
)


class OfflineSystemDemo:

    def __init__(self):

        checkpoint = (
            Path(MULTI_EXIT_CHECKPOINT_DIR)
            / MULTI_EXIT_BEST_MODEL_NAME
        )

        self.inference = MultiExitInference(
            checkpoint_path=checkpoint
        )

        self.dataset = TrafficDataset(
            dataset_root=DATASET_ROOT,
            verbose=False,
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
        )

        self.index_to_label = self.dataset.index_to_label

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------

        self.total = 0

        self.correct = 0

        self.exit_counter = defaultdict(int)

        self.class_correct = defaultdict(int)

        self.class_total = defaultdict(int)

        self.class_predictions = defaultdict(int)

        self.total_latency = 0.0

        self.total_packets = 0
    
        # ---------------------------------------------------------
    # Run Evaluation
    # ---------------------------------------------------------

    def run(self):

        print("=" * 90)
        print("OFFLINE SYSTEM DEMONSTRATION")
        print("=" * 90)

        print()

        print("Dataset Size :", len(self.dataset))

        print()

        print("Running Evaluation...")

        print()

        for index, (sequence, label) in enumerate(self.loader):

            sequence = sequence.squeeze(0)

            ground_truth = label.item()

            start = time.perf_counter()

            result = self.inference.predict(

                sequence,

                threshold=0.70,

            )

            end = time.perf_counter()

            latency = end - start

            self.total_latency += latency

            predicted = result["prediction"]

            exit_used = result["exit"]

            confidence = result["confidence"]

            self.total += 1

            self.exit_counter[exit_used] += 1

            packets_used = {

                "exit1": 5,

                "exit2": 10,

                "exit3": 15,

                "final": 20,

            }[exit_used]

            self.total_packets += packets_used

            gt_name = self.index_to_label[ground_truth]

            pred_name = self.index_to_label[predicted]

            self.class_total[gt_name] += 1

            self.class_predictions[pred_name] += 1

            if predicted == ground_truth:

                self.correct += 1

                self.class_correct[gt_name] += 1

            # -------------------------------------------------
            # Display First 20 Samples
            # -------------------------------------------------

            if index < 20:

                print("-" * 70)

                print(f"Sample {index + 1}")

                print()

                print(f"Ground Truth : {gt_name}")

                print(f"Prediction   : {pred_name}")

                print(f"Confidence   : {confidence:.2%}")

                print(f"Exit Used    : {exit_used}")

                print(f"Inference    : {latency*1000:.3f} ms")

                if predicted == ground_truth:

                    print("Result       : CORRECT")

                else:

                    print("Result       : WRONG")

        print()

        print("=" * 90)

        print("EVALUATION FINISHED")

        print("=" * 90)

        print()

        self.print_summary()

        # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def print_summary(self):

        accuracy = self.correct / self.total

        avg_latency = self.total_latency / self.total

        avg_packets = self.total_packets / self.total

        reduction = (

            (20 - avg_packets)

            / 20

            * 100

        )

        print("=" * 90)

        print("OVERALL RESULTS")

        print("=" * 90)

        print()

        print(f"Samples Evaluated : {self.total}")

        print(f"Correct           : {self.correct}")

        print(f"Accuracy          : {accuracy:.2%}")

        print(f"Average Latency   : {avg_latency*1000:.3f} ms")

        print(f"Average Packets   : {avg_packets:.2f} / 20")

        print(f"Latency Reduction : {reduction:.2f}%")

        print()

        print("=" * 90)

        print("EXIT DISTRIBUTION")

        print("=" * 90)

        for exit_name in [

            "exit1",

            "exit2",

            "exit3",

            "final",

        ]:

            count = self.exit_counter[exit_name]

            percentage = 100 * count / self.total

            print(

                f"{exit_name:<10}"

                f"{count:>6}"

                f"   "

                f"{percentage:6.2f}%"

            )

        print()

        print("=" * 90)

        print("PER-CLASS ACCURACY")

        print("=" * 90)

        for class_name in sorted(self.class_total.keys()):

            total = self.class_total[class_name]

            correct = self.class_correct[class_name]

            predicted = self.class_predictions[class_name]

            class_acc = (

                100 * correct / total

                if total

                else 0

            )

            print(

                f"{class_name:<15}"

                f"Accuracy : {class_acc:6.2f}%"

                f"   "

                f"Samples : {total:5d}"

                f"   "

                f"Predicted : {predicted:5d}"

            )

        print()

        print("=" * 90)
        # ---------------------------------------------------------
    # Confusion Summary
    # ---------------------------------------------------------

    def print_prediction_distribution(self):

        print()

        print("=" * 90)
        print("PREDICTION DISTRIBUTION")
        print("=" * 90)

        total_predictions = sum(
            self.class_predictions.values()
        )

        for class_name in sorted(self.class_predictions.keys()):

            count = self.class_predictions[class_name]

            percentage = (
                100 * count / total_predictions
                if total_predictions
                else 0
            )

            print(
                f"{class_name:<15}"
                f"{count:>6}"
                f"   "
                f"{percentage:6.2f}%"
            )

        print()

        print("=" * 90)

        self.print_prediction_distribution()

# =============================================================================
# Entry Point
# =============================================================================

def main():

    demo = OfflineSystemDemo()

    demo.run()


if __name__ == "__main__":

    main()