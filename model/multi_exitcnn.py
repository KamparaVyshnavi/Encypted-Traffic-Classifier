"""
===============================================================================
Multi-Exit Temporal CNN
===============================================================================
Description
-----------
Implements the Multi-Exit Temporal CNN architecture.

Instead of producing a prediction only after the final convolution layer,
this network attaches classifier heads at intermediate stages, allowing
future early-exit inference.

Architecture
------------
Input
    ↓
Conv Block 1
    ├── Exit 1
    ↓
Conv Block 2
    ├── Exit 2
    ↓
Conv Block 3
    ├── Exit 3
    ↓
Adaptive Average Pooling
    ↓
Final Classifier

Current Configuration
---------------------
Sequence Length : 20
Feature Dimension : 6
Number of Classes : 6

Future Extensions
-----------------
• Confidence-based Early Exit
• Latency-aware Inference
• Adaptive Exit Thresholds
• Dynamic Computation

"""

from __future__ import annotations

from typing import Dict
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from model.temporal_cnn import ConvBlock
from model.temporal_cnn import initialize_weights

from utils.config import DEFAULT_SEQUENCE_LENGTH

from utils.config import (
    FEATURE_DIMENSION,
    NUM_CLASSES,
    CNN_CHANNELS,
    FC_HIDDEN,
    FC_DROPOUT,
)


# =============================================================================
# Exit Head
# =============================================================================


class ExitHead(nn.Module):
    """
    Classification head attached to an intermediate feature map.

    Structure
    ---------

    AdaptiveAvgPool1d
            ↓
        Flatten
            ↓
        Linear
            ↓
          ReLU
            ↓
        Dropout
            ↓
        Linear
            ↓
        Class Logits
    """

    def __init__(
        self,
        input_channels: int,
    ) -> None:

        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(

            nn.Linear(
                input_channels,
                FC_HIDDEN,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                FC_DROPOUT,
            ),

            nn.Linear(
                FC_HIDDEN,
                NUM_CLASSES,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.global_pool(x)

        x = x.squeeze(-1)

        logits = self.classifier(x)

        return logits


# =============================================================================
# Multi Exit CNN
# =============================================================================


class MultiExitCNN(nn.Module):
    """
    Multi-Exit Temporal CNN.

    Input Shape
    -----------

    (batch_size,
     sequence_length,
     feature_dimension)

    Example
    -------

    (64, 20, 6)

    Notes
    -----

    Three intermediate exits are attached after each convolution block.

    A final classifier predicts using the complete feature representation.
    """

    def __init__(self) -> None:

        super().__init__()

        # ------------------------------------------------------------------
        # Shared Temporal Backbone
        # ------------------------------------------------------------------

        self.block1 = ConvBlock(
            FEATURE_DIMENSION,
            CNN_CHANNELS[0],
        )

        self.block2 = ConvBlock(
            CNN_CHANNELS[0],
            CNN_CHANNELS[1],
        )

        self.block3 = ConvBlock(
            CNN_CHANNELS[1],
            CNN_CHANNELS[2],
        )

        # ------------------------------------------------------------------
        # Exit Heads
        # ------------------------------------------------------------------

        self.exit1 = ExitHead(
            CNN_CHANNELS[0],
        )

        self.exit2 = ExitHead(
            CNN_CHANNELS[1],
        )

        self.exit3 = ExitHead(
            CNN_CHANNELS[2],
        )

        # ------------------------------------------------------------------
        # Final Pooling
        # ------------------------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ------------------------------------------------------------------
        # Final Classifier
        # ------------------------------------------------------------------

        self.final_classifier = nn.Sequential(

            nn.Linear(
                CNN_CHANNELS[2],
                FC_HIDDEN,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                FC_DROPOUT,
            ),

            nn.Linear(
                FC_HIDDEN,
                NUM_CLASSES,
            ),
        )

        # ------------------------------------------------------------------
        # Feature Cache
        # ------------------------------------------------------------------

        self.feature_cache: Dict[
            str,
            torch.Tensor,
        ] = {}

        # ------------------------------------------------------------------
        # Initialize Parameters
        # ------------------------------------------------------------------

        self.apply(
            initialize_weights,
        )
    
    def forward(
    self,
    x: torch.Tensor,
    return_features: bool = False,
) -> Union[
    Dict[str, torch.Tensor],
    Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ],
]:
        """
        Forward pass.

        Parameters
        ----------
        x
            Shape:
                (batch_size, sequence_length, feature_dimension)

        return_features
            If True, intermediate feature maps are also returned.

        Returns
        -------
        outputs

        {
            "exit1": logits,
            "exit2": logits,
            "exit3": logits,
            "final": logits,
        }

        or

        outputs, feature_dictionary
        """

        # ------------------------------------------------------------
        # Input Validation
        # ------------------------------------------------------------

        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input tensor "
                f"(batch, sequence, features), "
                f"got shape {tuple(x.shape)}"
            )

        if x.shape[-1] != FEATURE_DIMENSION:
            raise ValueError(
                f"Expected feature dimension "
                f"{FEATURE_DIMENSION}, "
                f"got {x.shape[-1]}"
            )

        # ------------------------------------------------------------
        # Conv1D expects:
        # (batch, channels, sequence)
        # ------------------------------------------------------------

        x = x.permute(0, 2, 1)

        # ------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------

        block1 = self.block1(x)
        exit1_logits = self.exit1(block1)

        block2 = self.block2(block1)
        exit2_logits = self.exit2(block2)

        block3 = self.block3(block2)
        exit3_logits = self.exit3(block3)

        # ------------------------------------------------------------
        # Final Prediction
        # ------------------------------------------------------------

        pooled = self.global_pool(block3)

        embedding = pooled.squeeze(-1)

        final_logits = self.final_classifier(embedding)

        # ------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------

        outputs = {

    "logits": {

        "exit1": exit1_logits,

        "exit2": exit2_logits,

        "exit3": exit3_logits,

        "final": final_logits,
    },

    "embedding": embedding,
}

        # ------------------------------------------------------------
        # Feature Cache
        # ------------------------------------------------------------

        self.feature_cache = {

            "block1": block1,

            "block2": block2,

            "block3": block3,

            "embedding": embedding,

            "exit1_logits": exit1_logits,

            "exit2_logits": exit2_logits,

            "exit3_logits": exit3_logits,

            "final_logits": final_logits,
        }

        # ------------------------------------------------------------
        # Optional Return
        # ------------------------------------------------------------

        if return_features:
            return outputs, self.feature_cache

        return outputs

    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the final embedding before classification.
        """

        _, features = self.forward(
            x,
            return_features=True,
        )

        return features["embedding"]

    def get_feature_cache(
        self,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns cached feature maps from the most recent
        forward pass.
        """

        if not self.feature_cache:
            raise RuntimeError(
                "Feature cache is empty. "
                "Run a forward pass before requesting features."
            )

        return self.feature_cache

    def count_parameters(
        self,
        trainable_only: bool = True,
    ) -> int:
        """
        Counts model parameters.
        """

        if trainable_only:

            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    def model_info(
        self,
    ) -> None:
        """
        Prints a concise model summary.
        """

        print("=" * 60)
        print("Multi-Exit Temporal CNN")
        print("=" * 60)

        print(f"Input Features     : {FEATURE_DIMENSION}")
        print(f"Sequence Length    : {DEFAULT_SEQUENCE_LENGTH}")
        print(f"Number of Classes  : {NUM_CLASSES}")
        print(f"CNN Channels       : {CNN_CHANNELS}")
        print(f"Hidden Dimension   : {FC_HIDDEN}")
        print("Exit Heads         : 3 Intermediate + 1 Final")

        print(
            f"Trainable Params   : "
            f"{self.count_parameters():,}"
        )

        print("=" * 60)


if __name__ == "__main__":

    model = MultiExitCNN()

    model.model_info()

    dummy = torch.randn(
        4,
        DEFAULT_SEQUENCE_LENGTH,
        FEATURE_DIMENSION,
    )

    outputs = model(dummy)

    print()

    for name, logits in outputs.items():
        print(f"{name:<10}: {tuple(logits.shape)}")

    embedding = model.extract_features(dummy)

    print()
    print("Embedding Shape :", embedding.shape)