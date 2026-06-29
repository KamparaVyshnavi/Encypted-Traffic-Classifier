"""
===============================================================================
Temporal CNN Backbone
===============================================================================

File        : temporal_cnn.py
Project     : Encrypted Network Traffic Classifier

Description
-----------
Implements the baseline Temporal Convolutional Neural Network (Temporal CNN)
used for encrypted traffic classification.

The network learns temporal relationships among the first N packets of every
network flow using 1D convolutions.

Architecture
------------
Input
    ↓
Conv Block 1
    ↓
Conv Block 2
    ↓
Conv Block 3
    ↓
Adaptive Average Pooling
    ↓
Fully Connected Classifier
    ↓
Class Logits

Current Configuration
---------------------
Sequence Length : 20
Feature Dimension : 6
Number of Classes : 6

Future Extensions
-----------------
Designed to support:

• Multi-Exit Inference
• Feature Visualization
• Embedding Extraction
• Ablation Studies
• Live Inference Pipeline

Author : Your Name
"""

from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import DEFAULT_SEQUENCE_LENGTH

from utils.config import (
    FEATURE_DIMENSION,
    NUM_CLASSES,
    CNN_CHANNELS,
    KERNEL_SIZE,
    STRIDE,
    PADDING,
    CONV_DROPOUT,
    FC_HIDDEN,
    FC_DROPOUT,
)

def initialize_weights(module: nn.Module) -> None:
    """
    Initializes network weights.

    Conv1D
        Kaiming Normal Initialization

    BatchNorm
        Weight = 1
        Bias = 0

    Linear
        Xavier Uniform Initialization
    """

    if isinstance(module, nn.Conv1d):

        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.BatchNorm1d):

        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):

        nn.init.xavier_uniform_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ConvBlock(nn.Module):
    """
    Standard Temporal CNN Block.

    Structure
    ---------
    Conv1D
        ↓
    BatchNorm
        ↓
    ReLU
        ↓
    Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            bias=True,
        )

        self.batch_norm = nn.BatchNorm1d(out_channels)

        self.activation = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(CONV_DROPOUT)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv(x)

        x = self.batch_norm(x)

        x = self.activation(x)

        x = self.dropout(x)

        return x
    
class TemporalCNN(nn.Module):
    """
    Baseline Temporal Convolutional Neural Network.

    Input Shape
    -----------
    (batch_size, sequence_length, feature_dimension)

    Example
    -------
    (64, 20, 6)

    Internal Shape
    --------------
    Conv1D expects

    (batch_size, channels, sequence_length)

    therefore the input is transposed before feature extraction.
    """

    def __init__(self) -> None:

        super().__init__()

        # ------------------------------------------------------------------
        # Feature Extraction Backbone
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
        # Global Feature Aggregation
        # ------------------------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ------------------------------------------------------------------
        # Classifier Head
        # ------------------------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                CNN_CHANNELS[2],
                FC_HIDDEN,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(FC_DROPOUT),

            nn.Linear(
                FC_HIDDEN,
                NUM_CLASSES,
            )
        )

        # ------------------------------------------------------------------
        # Feature Cache
        # ------------------------------------------------------------------

        self.feature_cache: Dict[str, torch.Tensor] = {}

        # ------------------------------------------------------------------
        # Initialize Parameters
        # ------------------------------------------------------------------

        self.apply(initialize_weights)

    
    from typing import Dict, Tuple, Union

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
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
        logits

        or

        logits, feature_dictionary
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

        block2 = self.block2(block1)

        block3 = self.block3(block2)

        # ------------------------------------------------------------
        # Global Feature Extraction
        # ------------------------------------------------------------

        pooled = self.global_pool(block3)

        embedding = pooled.squeeze(-1)

        logits = self.classifier(embedding)

        # ------------------------------------------------------------
        # Store Feature Cache
        # ------------------------------------------------------------

        self.feature_cache = {

            "block1": block1,

            "block2": block2,

            "block3": block3,

            "embedding": embedding,
        }

        # ------------------------------------------------------------
        # Optional Return
        # ------------------------------------------------------------

        if return_features:
            return logits, self.feature_cache

        return logits
    
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
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns predicted class indices.

        Shape
        -----

        (batch_size,)
        """

        self.eval()

        logits = self.forward(x)

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        return predictions
    

    def get_feature_cache(self) -> Dict[str, torch.Tensor]:
        """
        Returns the cached intermediate feature maps from the
        most recent forward pass.

        Returns
        -------
        Dict[str, Tensor]

            block1
            block2
            block3
            embedding
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

        Parameters
        ----------
        trainable_only
            If True, counts only trainable parameters.

        Returns
        -------
        int
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
    
    def model_info(self) -> None:
        """
        Prints a concise summary of the architecture.
        """

        print("=" * 60)
        print("Temporal CNN")
        print("=" * 60)

        print(f"Input Features     : {FEATURE_DIMENSION}")
        print(f"Sequence Length    : {DEFAULT_SEQUENCE_LENGTH}")
        print(f"Number of Classes  : {NUM_CLASSES}")
        print(f"CNN Channels       : {CNN_CHANNELS}")
        print(f"Hidden Dimension   : {FC_HIDDEN}")

        print(
            f"Trainable Params   : "
            f"{self.count_parameters():,}"
        )

        print("=" * 60)

if __name__ == "__main__":

    model = TemporalCNN()

    model.model_info()

    dummy = torch.randn(
        4,
        20,
        FEATURE_DIMENSION,
    )

    logits = model(dummy)

    print("Output Shape :", logits.shape)

    prediction = model.predict(dummy)

    print("Prediction Shape :", prediction.shape)

    embedding = model.extract_features(dummy)

    print("Embedding Shape :", embedding.shape)
