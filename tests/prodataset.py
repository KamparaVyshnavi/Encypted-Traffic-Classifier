"""
=========================================================================
Test : Traffic Dataset
=========================================================================
"""

import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Add Project Root
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

from model.dataset import TrafficDataset
from utils.config import (
    DEFAULT_SEQUENCE_LENGTH,
    FEATURE_DIMENSION,
    NUM_CLASSES,
)

# ----------------------------------------------------------------------
# Main Test
# ----------------------------------------------------------------------


def main():

    print("=" * 70)
    print("TRAFFIC DATASET TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------------

    print("\n[1] Loading Dataset...")

    dataset = TrafficDataset()

    print("✓ Dataset loaded successfully.")

    # ------------------------------------------------------------
    # Dataset Information
    # ------------------------------------------------------------

    print("\n[2] Dataset Information")

    dataset.dataset_info()

    # ------------------------------------------------------------
    # Dataset Length
    # ------------------------------------------------------------

    print("\n[3] Dataset Length")

    print("Samples :", len(dataset))

    assert len(dataset) > 0

    print("✓ Dataset length verified.")

    # ------------------------------------------------------------
    # Load First Sample
    # ------------------------------------------------------------

    print("\n[4] Loading First Sample...")

    sequence, label = dataset[0]

    print("Sequence Shape :", tuple(sequence.shape))
    print("Sequence Type  :", sequence.dtype)

    print("Label          :", label.item())
    print("Label Type     :", label.dtype)

    assert sequence.shape == (
        DEFAULT_SEQUENCE_LENGTH,
        FEATURE_DIMENSION,
    )

    assert sequence.dtype == torch.float32

    assert label.dtype == torch.long

    assert 0 <= label.item() < dataset.num_classes()

    print("✓ First sample verified.")

    # ------------------------------------------------------------
    # Random Sample
    # ------------------------------------------------------------

    print("\n[5] Loading Random Sample...")

    index = len(dataset) // 2

    sequence, label = dataset[index]

    print("Index :", index)

    print("Sequence Shape :", tuple(sequence.shape))

    print("Label :", label.item())

    assert sequence.shape == (
        DEFAULT_SEQUENCE_LENGTH,
        FEATURE_DIMENSION,
    )

    print("✓ Random sample verified.")

    # ------------------------------------------------------------
    # Class Distribution
    # ------------------------------------------------------------

    print("\n[6] Class Distribution")

    distribution = dataset.get_class_distribution()

    total = sum(distribution.values())

    print(distribution)

    print("Total Samples :", total)

    assert total == len(dataset)

    print("✓ Distribution verified.")

    # ------------------------------------------------------------
    # Tensor Checks
    # ------------------------------------------------------------

    print("\n[7] Tensor Validation")

    print("Min :", torch.min(sequence).item())
    print("Max :", torch.max(sequence).item())

    assert torch.isfinite(sequence).all()

    print("✓ Tensor validation passed.")

    # ------------------------------------------------------------
    # Success
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL DATASET TESTS PASSED")
    print("=" * 70)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()