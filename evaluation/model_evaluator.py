"""
===============================================================================
Model Evaluator
===============================================================================

Generic evaluation engine for trained encrypted traffic classifiers.

Responsibilities
----------------
1. Load a trained checkpoint.
2. Load any processed dataset.
3. Run inference only.
4. Collect predictions.
5. Compute evaluation metrics.
6. Return results to higher-level evaluation scripts.

This file DOES NOT train models.

Future Uses
-----------
- Cross-Network Evaluation
- Ablation Study
- Multi-Exit Evaluation
- Model Comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from model.dataset import TrafficDataset
from model.temporal_cnn import TemporalCNN
from model.inference import MultiExitInference

from utils.config import (
    DEVICE,
    BATCH_SIZE,
    PIN_MEMORY,
    NON_BLOCKING,
)


# =============================================================================
# Evaluation Result
# =============================================================================

@dataclass
class EvaluationResult:

    accuracy: float

    precision: float

    recall: float

    f1_score: float

    confusion_matrix: np.ndarray

    classification_report: str

    predictions: List[int]

    targets: List[int]


# =============================================================================
# Model Evaluator
# =============================================================================

class ModelEvaluator:

    """
    Generic inference engine.

    Example
    -------

    evaluator = ModelEvaluator()

    result = evaluator.evaluate(
        checkpoint_path,
        dataset_root,
    )
    """

    def __init__(
    self,
    multi_exit: bool = False,
    threshold: float = 0.90,
):

        self.multi_exit = multi_exit

        self.threshold = threshold

        self.model = None

        self.dataset = None

        self.dataloader = None

        self.inference = None

        self.exit_counts = {

            "exit1": 0,

            "exit2": 0,

            "exit3": 0,

            "final": 0,
        }

        self.total_blocks = 0


    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    def load_model(
        self,
        checkpoint_path: str | Path,
    ):

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"{checkpoint_path}"
            )
        if self.multi_exit:

            self.inference = MultiExitInference(
                checkpoint_path,
            )

            return

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

        model = TemporalCNN()

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(DEVICE)

        model.eval()

        self.model = model


    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    def load_dataset(
        self,
        dataset_root: str | Path,
    ):

        dataset = TrafficDataset(
            dataset_root=dataset_root,
            verbose=False,
        )

        dataloader = DataLoader(

            dataset,

            batch_size=BATCH_SIZE,

            shuffle=False,

            pin_memory=PIN_MEMORY,

            num_workers=0,
        )

        self.dataset = dataset

        self.dataloader = dataloader


    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def predict(self):

        predictions = []

        targets = []

        if self.multi_exit:

            for sequences, labels in self.dataloader:

                for sequence, label in zip(
                    sequences,
                    labels,
                ):

                    result = self.inference.predict(

                        sequence,

                        threshold=self.threshold,
                    )

                    predictions.append(
                        result["prediction"]
                    )

                    targets.append(
                        label.item()
                    )

                    self.exit_counts[
                        result["exit"]
                    ] += 1

                    self.total_blocks += (
                        result["blocks_executed"]
                    )

            return predictions, targets

        # -------------------------------------------------
        # Original CNN
        # -------------------------------------------------

        for sequences, labels in self.dataloader:

            sequences = sequences.to(
                DEVICE,
                non_blocking=NON_BLOCKING,
            )

            logits = self.model(sequences)

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            predictions.extend(
                predicted.cpu().numpy().tolist()
            )

            targets.extend(
                labels.numpy().tolist()
            )

        return predictions, targets
        # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def compute_metrics(
        self,
        predictions,
        targets,
    ) -> EvaluationResult:

        accuracy = accuracy_score(
            targets,
            predictions,
        )

        precision, recall, f1_score, _ = (
            precision_recall_fscore_support(
                targets,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )

        # ----------------------------------------------------------
        # Evaluate only labels present in the dataset
        # ----------------------------------------------------------

        labels = sorted(
            set(targets)
        )

        target_names = [

            self.dataset.index_to_label[label]

            for label in labels

        ]

        cm = confusion_matrix(

            targets,

            predictions,

            labels=labels,

        )

        report = classification_report(

            targets,

            predictions,

            labels=labels,

            target_names=target_names,

            digits=4,

            zero_division=0,

        )

        return EvaluationResult(

            accuracy=accuracy,

            precision=precision,

            recall=recall,

            f1_score=f1_score,

            confusion_matrix=cm,

            classification_report=report,

            predictions=predictions,

            targets=targets,
        )


    # -------------------------------------------------------------------------
    # Pretty Printing
    # -------------------------------------------------------------------------

    def print_results(
        self,
        result: EvaluationResult,
    ) -> None:

        print()

        print("=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

        print()

        print(
            f"Accuracy  : "
            f"{result.accuracy:.4%}"
        )

        print(
            f"Precision : "
            f"{result.precision:.4%}"
        )

        print(
            f"Recall    : "
            f"{result.recall:.4%}"
        )

        print(
            f"F1 Score  : "
            f"{result.f1_score:.4%}"
        )
        if self.multi_exit:

            print()

            print("=" * 70)

            print("EARLY EXIT STATISTICS")

            print("=" * 70)

            total = sum(
                self.exit_counts.values()
            )

            for exit_name, count in self.exit_counts.items():

                print(

                    f"{exit_name:<8}"

                    f": "

                    f"{count:5d}"

                    f" ({count/total:.2%})"

                )

            print()

            print(

                f"Average Blocks : "

                f"{self.total_blocks/total:.2f}"

            )

        print()

        print("=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)

        print(result.classification_report)

        print("=" * 70)

        print()

        print("CONFUSION MATRIX")

        print()

        print(result.confusion_matrix)

        print()

        # -------------------------------------------------------------------------
    # Complete Evaluation Pipeline
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        checkpoint_path: str | Path,
        dataset_root: str | Path,
        print_summary: bool = True,
    ) -> EvaluationResult:

        """
        Complete evaluation pipeline.

        Steps
        -----
        1. Load trained model
        2. Load dataset
        3. Perform inference
        4. Compute metrics
        5. Print results (optional)
        6. Return EvaluationResult
        """

        self.load_model(
            checkpoint_path
        )

        self.load_dataset(
            dataset_root
        )

        predictions, targets = self.predict()

        result = self.compute_metrics(
            predictions,
            targets,
        )

        if print_summary:

            self.print_results(
                result
            )
        result.evaluator = self
        return result


# =============================================================================
# Convenience Function
# =============================================================================

def evaluate_model(
    checkpoint_path,
    dataset_root,
    print_summary=True,
    multi_exit=False,
    threshold=0.90,
):
    """
    Functional interface.

    Example
    -------
    result = evaluate_model(
        checkpoint,
        dataset,
    )
    """

    evaluator = ModelEvaluator(

    multi_exit=multi_exit,

    threshold=threshold,
)

    return evaluator.evaluate(
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        print_summary=print_summary,
    )


# =============================================================================
# Standalone Execution
# =============================================================================

if __name__ == "__main__":

    from utils.config import (
        CHECKPOINT_DIR,
        BEST_MODEL_NAME,
        DATASET_ROOT,
    )

    checkpoint = (
        Path(CHECKPOINT_DIR)
        /
        BEST_MODEL_NAME
    )

    evaluate_model(
        checkpoint,
        DATASET_ROOT,
    )