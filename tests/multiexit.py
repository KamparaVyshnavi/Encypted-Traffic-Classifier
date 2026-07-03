"""
===============================================================================
Multi-Exit CNN Test
===============================================================================

Verifies:

✓ Model construction
✓ Parameter count
✓ Forward pass
✓ Output shapes
✓ Feature extraction
✓ Feature cache
✓ Gradient flow from every exit
✓ CUDA compatibility (if available)

Author : Your Name
"""

import torch

from model.multi_exitcnn import MultiExitCNN

from utils.config import (
    FEATURE_DIMENSION,
    DEFAULT_SEQUENCE_LENGTH,
    NUM_CLASSES,
    CNN_CHANNELS,
)


def separator():
    print("=" * 70)


def check(condition, message):
    if condition:
        print(f"✓ {message}")
    else:
        raise AssertionError(f"✗ {message}")


def main():

    separator()
    print("MULTI-EXIT CNN TEST")
    separator()

    # ------------------------------------------------------------------
    # Model Creation
    # ------------------------------------------------------------------

    model = MultiExitCNN()

    check(True, "Model created successfully")

    print(f"\nTrainable Parameters : {model.count_parameters():,}")

    model.model_info()

    # ------------------------------------------------------------------
    # Dummy Input
    # ------------------------------------------------------------------

    batch_size = 4

    dummy = torch.randn(
        batch_size,
        DEFAULT_SEQUENCE_LENGTH,
        FEATURE_DIMENSION,
    )

    print("\nInput Shape :", tuple(dummy.shape))

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    outputs = model(dummy)

    check(isinstance(outputs, dict), "Forward returns dictionary")

    check("logits" in outputs, "Logits dictionary exists")
    check("embedding" in outputs, "Embedding exists")

    logits = outputs["logits"]

    required_outputs = [
        "exit1",
        "exit2",
        "exit3",
        "final",
    ]

    print("\nOutput Shapes")

    for key in required_outputs:

        check(key in logits, f"{key} exists")

        expected = (batch_size, NUM_CLASSES)

        actual = tuple(logits[key].shape)

        check(
            actual == expected,
            f"{key} shape = {expected}",
        )

        print(f"   {key:<8}: {actual}")

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    embedding = model.extract_features(dummy)
    check(
    outputs["embedding"].shape == (batch_size, CNN_CHANNELS[2]),
    "Forward embedding shape correct",
)
    expected_embedding = (
        batch_size,
        CNN_CHANNELS[2],
    )

    check(
        tuple(embedding.shape) == expected_embedding,
        "Embedding shape correct",
    )

    print("\nEmbedding Shape :", tuple(embedding.shape))

    # ------------------------------------------------------------------
    # Feature Cache
    # ------------------------------------------------------------------

    cache = model.get_feature_cache()

    required_cache = [

        "block1",

        "block2",

        "block3",

        "embedding",

        "exit1_logits",

        "exit2_logits",

        "exit3_logits",

        "final_logits",
    ]

    print("\nFeature Cache")

    for key in required_cache:

        check(key in cache, f"{key} cached")

    print()

    print("block1 :", tuple(cache["block1"].shape))
    print("block2 :", tuple(cache["block2"].shape))
    print("block3 :", tuple(cache["block3"].shape))
    print("embedding :", tuple(cache["embedding"].shape))

    # ------------------------------------------------------------------
    # Gradient Flow
    # ------------------------------------------------------------------

    print("\nGradient Flow Test")

    exit_names = [
        "exit1",
        "exit2",
        "exit3",
        "final",
    ]

    for exit_name in exit_names:

        model.zero_grad()

        outputs = model(dummy)

        loss = outputs["logits"][exit_name].mean()

        loss.backward()

        has_gradient = any(

            parameter.grad is not None

            for parameter in model.parameters()

            if parameter.requires_grad

        )
        

        check(
            has_gradient,
            f"{exit_name} backward pass",
        )

    # ------------------------------------------------------------------
    # CUDA Test
    # ------------------------------------------------------------------

    print("\nCUDA Compatibility")

    if torch.cuda.is_available():

        device = torch.device("cuda")

        model = model.to(device)

        dummy_gpu = dummy.to(device)

        outputs = model(dummy_gpu)

        for key in required_outputs:

            check(
                outputs["logits"][key].is_cuda,
                f"{key} on CUDA",
            )

        print("✓ CUDA forward pass successful")

    else:

        print("CUDA not available - skipped")

    # ------------------------------------------------------------------
    # Prediction Consistency
    # ------------------------------------------------------------------

    print("\nPrediction Shapes")

    outputs = model(dummy if not torch.cuda.is_available() else dummy_gpu)

    for key in required_outputs:

        predictions = torch.argmax(outputs["logits"][key], dim=1)

        check(
            predictions.shape == (batch_size,),
            f"{key} predictions",
        )

    # ------------------------------------------------------------------
    # Numerical Stability
    # ------------------------------------------------------------------

    print("\nNumerical Stability")

    for key in required_outputs:

        tensor = outputs["logits"][key]

        check(
            not torch.isnan(tensor).any(),
            f"{key} contains no NaNs",
        )

        check(
            not torch.isinf(tensor).any(),
            f"{key} contains no Infs",
        )

    separator()
    print("ALL MULTI-EXIT TESTS PASSED SUCCESSFULLY")
    separator()


if __name__ == "__main__":

    main()