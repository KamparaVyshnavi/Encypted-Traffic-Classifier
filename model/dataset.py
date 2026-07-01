"""
===============================================================================
Traffic Dataset
===============================================================================

Loads processed encrypted traffic sequences generated during preprocessing.

Dataset Structure
-----------------

datasets/
└── processed_sequences/
    ├── sequences/
    │      sample_0000000.npy
    │      ...
    ├── labels.csv
    └── metadata.json

labels.csv

sample,label

sample_0000000,Chat
sample_0000001,Streaming
...

Unknown samples are excluded from training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.config import (
    DEFAULT_SEQUENCE_LENGTH,
    FEATURE_DIMENSION,
    TENSOR_DTYPE,
)


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for encrypted traffic classification.
    """

    def __init__(
        self,
        dataset_root: str | Path = "datasets/processed_sequences",
        verbose: bool = True,
    ) -> None:

        super().__init__()

        self.dataset_root = Path(dataset_root)

        self.sequence_directory = self.dataset_root / "sequences"
        self.label_file = self.dataset_root / "labels.csv"

        if not self.sequence_directory.exists():
            raise FileNotFoundError(
                f"Sequence directory not found:\n"
                f"{self.sequence_directory}"
            )

        if not self.label_file.exists():
            raise FileNotFoundError(
                f"Label file not found:\n"
                f"{self.label_file}"
            )

        self.labels = pd.read_csv(self.label_file)

        required_columns = {"sample", "label"}

        if not required_columns.issubset(self.labels.columns):
            raise ValueError(
                "labels.csv must contain columns:\n"
                "sample,label"
            )

        # ------------------------------------------------------------
        # Remove Unknown class
        # ------------------------------------------------------------

        original_samples = len(self.labels)

        self.labels = (
            self.labels[self.labels["label"] != "Unknown"]
            .reset_index(drop=True)
        )

        removed = original_samples - len(self.labels)

        if removed > 0 and verbose:
            print(f"Removed {removed} Unknown samples.")

        # ------------------------------------------------------------
        # Create Label Encoding
        # ------------------------------------------------------------

        unique_labels = sorted(self.labels["label"].unique())

        self.label_to_index = {
            label: index
            for index, label in enumerate(unique_labels)
        }

        self.index_to_label = {
            index: label
            for label, index in self.label_to_index.items()
        }

    def __len__(self) -> int:

        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        row = self.labels.iloc[index]

        sample_name = str(row["sample"]) + ".npy"

        sample_path = self.sequence_directory / sample_name

        if not sample_path.exists():
            raise FileNotFoundError(
                f"Missing sample:\n{sample_path}"
            )

        sequence = np.load(sample_path)

        if sequence.shape != (
            DEFAULT_SEQUENCE_LENGTH,
            FEATURE_DIMENSION,
        ):
            raise ValueError(
                f"Expected shape "
                f"({DEFAULT_SEQUENCE_LENGTH}, "
                f"{FEATURE_DIMENSION}) "
                f"but got {sequence.shape}"
            )

        sequence = torch.from_numpy(sequence).to(TENSOR_DTYPE)

        label_name = row["label"]

        label = torch.tensor(
            self.label_to_index[label_name],
            dtype=torch.long,
        )

        return sequence, label

    def num_classes(self) -> int:

        return len(self.label_to_index)

    def get_class_distribution(self) -> Dict[str, int]:

        return (
            self.labels["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    def dataset_info(self) -> None:

        print("=" * 60)
        print("Traffic Dataset")
        print("=" * 60)

        print(f"Samples            : {len(self)}")
        print(f"Sequence Length    : {DEFAULT_SEQUENCE_LENGTH}")
        print(f"Feature Dimension  : {FEATURE_DIMENSION}")
        print(f"Classes            : {self.num_classes()}")

        print("\nLabel Mapping")

        for label, index in self.label_to_index.items():
            print(f"{index} -> {label}")

        print("\nClass Distribution")

        distribution = self.get_class_distribution()

        for label, count in distribution.items():
            print(f"{label:<15} : {count}")

        print("=" * 60)