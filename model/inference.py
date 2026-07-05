"""
===============================================================================
Multi-Exit Inference Engine
===============================================================================

File        : inference.py

Description
-----------
Implements confidence-based inference for the trained
Multi-Exit Temporal CNN.

Part 1
------

✓ Load trained checkpoint
✓ Initialize model
✓ Softmax confidence
✓ Single-sample prediction

Future Parts
------------

Part 2
    Batch prediction

Part 3
    Dataset evaluation

Part 4
    Threshold sweep

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from model.multi_exitcnn import MultiExitCNN

from utils.config import (
    DEVICE,
    FEATURE_DIMENSION,
)


class MultiExitInference:
    """
    Confidence-based inference engine.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
    ) -> None:

        self.device = DEVICE

        self.model = MultiExitCNN()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    @staticmethod
    def softmax_confidence(
        logits: torch.Tensor,
    ) -> tuple[int, float]:
        """
        Computes prediction and confidence.

        Returns
        -------
        prediction
        confidence
        """

        probabilities = F.softmax(
            logits,
            dim=0,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=0,
        )

        return (
            prediction.item(),
            confidence.item(),
        )

    @torch.no_grad()
    def predict(
        self,
        sequence: torch.Tensor,
        threshold: float = 0.90,
    ) -> Dict:
        """
        Performs confidence-based inference on
        one sequence.

        Returns
        -------

        prediction

        confidence

        exit
        """

        if sequence.ndim != 2:

            raise ValueError(
                "Expected shape "
                "(sequence_length, feature_dimension)"
            )

        if sequence.shape[1] != FEATURE_DIMENSION:

            raise ValueError(
                "Invalid feature dimension."
            )

        sequence = sequence.unsqueeze(0)

        sequence = sequence.to(self.device)

        return self.model.early_exit_forward(
            sequence,
            threshold,
        )

        
    @torch.no_grad()
    def predict_batch(
        self,
        sequences: torch.Tensor,
        threshold: float = 0.90,
    ):
        """
        Predicts an entire batch.
        """

        results = []

        for sequence in sequences:

            result = self.predict(
                sequence,
                threshold,
            )

            results.append(result)

        return results
    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        threshold: float = 0.90,
    ):
        """
        Evaluates early-exit inference.
        """

        total = 0

        correct = 0

        exit_counts = {

            "exit1": 0,

            "exit2": 0,

            "exit3": 0,

            "final": 0,
        }

        for sequences, labels in dataloader:

            results = self.predict_batch(
                sequences,
                threshold,
            )

            for result, label in zip(results, labels):

                total += 1

                exit_counts[result["exit"]] += 1

                if result["prediction"] == label.item():

                    correct += 1

        accuracy = correct / total

        total = sum(exit_counts.values())

        average_exit = (

            1 * exit_counts["exit1"] +

            2 * exit_counts["exit2"] +

            3 * exit_counts["exit3"] +

            4 * exit_counts["final"]

        ) / total

        average_packets = (

            5 * exit_counts["exit1"] +

            10 * exit_counts["exit2"] +

            15 * exit_counts["exit3"] +

            20 * exit_counts["final"]

        ) / total

        latency_reduction = (

            20 - average_packets

        ) / 20

        statistics = {

            "threshold": threshold,

            "accuracy": accuracy,

            "exit_counts": exit_counts,

            "average_exit": average_exit,

            "average_packets": average_packets,

            "latency_reduction": latency_reduction,
        }

        return statistics
    
    @staticmethod
    def print_statistics(
        statistics,
    ):

        total = sum(
            statistics["exit_counts"].values()
        )

        print("=" * 70)

        print(
            f"Threshold : "
            f"{statistics['threshold']:.2f}"
        )

        print(
            f"Accuracy  : "
            f"{statistics['accuracy']:.4%}"
        )

        print()

        print("Exit Distribution")

        for exit_name, count in statistics["exit_counts"].items():

            percentage = count / total

            print(
                f"{exit_name:<6}"
                f": "
                f"{count:5d}"
                f" ({percentage:.2%})"
            )

        print("=" * 70)
        print()

        print(
            f"Average Exit          : "
            f"{statistics['average_exit']:.2f}"
        )

        print(
            f"Average Packets Used  : "
            f"{statistics['average_packets']:.2f}"
        )

        print(
            f"Estimated Latency Reduction : "
            f"{statistics['latency_reduction']:.2%}"
        )

    @torch.no_grad()
    def sweep_thresholds(
        self,
        dataloader,
        thresholds=None,
    ):
        """
        Evaluates multiple confidence thresholds.

        Parameters
        ----------
        dataloader

        thresholds
            List of confidence thresholds.

        Returns
        -------
        List of statistics.
        """

        if thresholds is None:

            thresholds = [
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                0.99,
            ]

        results = []

        print("=" * 90)
        print("THRESHOLD SWEEP")
        print("=" * 90)

        print(
            f"{'Threshold':<12}"
            f"{'Accuracy':<12}"
            f"{'Exit1':<10}"
            f"{'Exit2':<10}"
            f"{'Exit3':<10}"
            f"{'Final':<10}"
        )

        print("-" * 90)

        for threshold in thresholds:

            statistics = self.evaluate(
                dataloader,
                threshold,
            )

            results.append(statistics)

            total = sum(
                statistics["exit_counts"].values()
            )

            e1 = statistics["exit_counts"]["exit1"] / total
            e2 = statistics["exit_counts"]["exit2"] / total
            e3 = statistics["exit_counts"]["exit3"] / total
            ef = statistics["exit_counts"]["final"] / total

            print(
                f"{threshold:<12.2f}"
                f"{statistics['accuracy']:<12.4%}"
                f"{e1:<10.2%}"
                f"{e2:<10.2%}"
                f"{e3:<10.2%}"
                f"{ef:<10.2%}"
            )

        print("=" * 90)

        return results