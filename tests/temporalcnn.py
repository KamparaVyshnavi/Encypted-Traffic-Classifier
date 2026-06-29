"""
=========================================================================
Test : Temporal CNN
=========================================================================
"""

import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Add project root to Python path
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

from model.temporal_cnn import TemporalCNN
from utils.config import (
    DEFAULT_SEQUENCE_LENGTH,
    FEATURE_DIMENSION,
    NUM_CLASSES,
    CNN_CHANNELS,
)

# ----------------------------------------------------------------------
# Main Test
# ----------------------------------------------------------------------


def main():

    print("=" * 70)
    print("TEMPORAL CNN TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # Model Construction
    # ------------------------------------------------------------

    print("\n[1] Constructing Model...")

    model = TemporalCNN()

    print("✓ Model created successfully.")

    # ------------------------------------------------------------
    # Model Summary
    # ------------------------------------------------------------

    print("\n[2] Model Information")

    model.model_info()

    # ------------------------------------------------------------
    # Dummy Input
    # ------------------------------------------------------------

    print("\n[3] Creating Dummy Input...")

    batch_size = 4

    dummy = torch.randn(
        batch_size,
        DEFAULT_SEQUENCE_LENGTH,
        FEATURE_DIMENSION,
    )

    print("Input Shape :", tuple(dummy.shape))

    # ------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------

    print("\n[4] Testing Forward Pass...")

    logits = model(dummy)

    print("Output Shape :", tuple(logits.shape))

    assert logits.shape == (
        batch_size,
        NUM_CLASSES,
    )

    print("✓ Forward pass successful.")

    # ------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------

    print("\n[5] Testing Feature Extraction...")

    embedding = model.extract_features(dummy)

    print("Embedding Shape :", tuple(embedding.shape))

    assert embedding.shape == (
        batch_size,
        CNN_CHANNELS[2],
    )

    print("✓ Feature extraction successful.")

    # ------------------------------------------------------------
    # Intermediate Features
    # ------------------------------------------------------------

    print("\n[6] Testing Intermediate Features...")

    logits, features = model(
        dummy,
        return_features=True,
    )

    print("Block 1 :", tuple(features["block1"].shape))
    print("Block 2 :", tuple(features["block2"].shape))
    print("Block 3 :", tuple(features["block3"].shape))
    print("Embedding :", tuple(features["embedding"].shape))

    assert features["block1"].shape == (
        batch_size,
        CNN_CHANNELS[0],
        DEFAULT_SEQUENCE_LENGTH,
    )

    assert features["block2"].shape == (
        batch_size,
        CNN_CHANNELS[1],
        DEFAULT_SEQUENCE_LENGTH,
    )

    assert features["block3"].shape == (
        batch_size,
        CNN_CHANNELS[2],
        DEFAULT_SEQUENCE_LENGTH,
    )

    assert features["embedding"].shape == (
        batch_size,
        CNN_CHANNELS[2],
    )

    print("✓ Intermediate features successful.")

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    print("\n[7] Testing Prediction...")

    prediction = model.predict(dummy)

    print("Prediction Shape :", tuple(prediction.shape))
    print("Predictions :", prediction.tolist())

    assert prediction.shape == (batch_size,)

    assert torch.all(
        prediction >= 0
    )

    assert torch.all(
        prediction < NUM_CLASSES
    )

    print("✓ Prediction successful.")

    # ------------------------------------------------------------
    # Parameter Count
    # ------------------------------------------------------------

    print("\n[8] Parameter Count")

    total = model.count_parameters()

    print(f"Trainable Parameters : {total:,}")

    assert total > 0

    print("✓ Parameter counting successful.")

    # ------------------------------------------------------------
    # Feature Cache
    # ------------------------------------------------------------

    print("\n[9] Feature Cache")

    cache = model.get_feature_cache()

    print("Cached Keys :", list(cache.keys()))

    assert len(cache) == 4

    print("✓ Feature cache successful.")

    # ------------------------------------------------------------
    # Success
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL TEMPORAL CNN TESTS PASSED")
    print("=" * 70)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()